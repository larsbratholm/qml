! MIT License
!
! Copyright (c) 2016-2017 Anders Steen Christensen, Lars Andersen Bratholm
!
! Permission is hereby granted, free of charge, to any person obtaining a copy
! of this software and associated documentation files (the "Software"), to deal
! in the Software without restriction, including without limitation the rights
! to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
! copies of the Software, and to permit persons to whom the Software is
! furnished to do so, subject to the following conditions:
!
! The above copyright notice and this permission notice shall be included in all
! copies or substantial portions of the Software.
!
! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
! IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
! AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
! OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
! SOFTWARE.

module representations

    implicit none

contains

subroutine get_indices(natoms, nuclear_charges, type1, n, type1_indices)
    integer, intent(in) :: natoms
    integer, intent(in) :: type1
    integer, dimension(:), intent(in) :: nuclear_charges

    integer, intent(out) :: n
    integer, dimension(:), intent(out) :: type1_indices
    integer :: j

    !$OMP PARALLEL DO REDUCTION(+:n)
    do j = 1, natoms
        if (nuclear_charges(j) == type1) then
            ! this shouldn't be a race condition
            n = n + 1
            type1_indices(n) = j
        endif
    enddo
    !$OMP END PARALLEL DO

end subroutine get_indices

!subroutine OMP_init_lock(svar)
!    integer(kind = OMP_lock_kind), intent(out) :: svar
!end subroutine OMP_init_lock
!
!subroutine call OMP_set_lock(svar)
!    integer(kind = OMP_lock_kind), intent(inout) :: svar
!end subroutine call OMP_set_lock
!
!subroutine call OMP_unset_lock(svar)
!    integer(kind = OMP_lock_kind), intent(inout) :: svar
!end subroutine call OMP_unset_lock
!
!subroutine OMP_destroy_lock(svar)
!    integer(kind = OMP_lock_kind), intent(inout) :: svar
!end subroutine OMP_destroy_lock

end module representations

subroutine fgenerate_coulomb_matrix(atomic_charges, coordinates, nmax, cm)
    
    implicit none

    double precision, dimension(:), intent(in) :: atomic_charges
    double precision, dimension(:,:), intent(in) :: coordinates

    integer, intent(in) :: nmax

    double precision, dimension(((nmax + 1) * nmax) / 2), intent(out):: cm

    double precision, allocatable, dimension(:) :: row_norms
    double precision :: pair_norm
    double precision :: huge_double

    integer, allocatable, dimension(:) :: sorted_atoms

    double precision, allocatable, dimension(:,:) :: pair_distance_matrix

    integer :: i, j, m, n, idx
    integer :: natoms


    if (size(coordinates, dim=1) /= size(atomic_charges, dim=1)) then
        write(*,*) "ERROR: Coulomb matrix generation"
        write(*,*) size(coordinates, dim=1), "coordinates, but", &
            & size(atomic_charges, dim=1), "atom_types!"
        stop
    else
        natoms = size(atomic_charges, dim=1)
    endif

    ! Allocate temporary
    allocate(pair_distance_matrix(natoms,natoms))
    allocate(row_norms(natoms))
    allocate(sorted_atoms(natoms))

    huge_double = huge(row_norms(1))

    ! Calculate row-2-norms and store pair-distances in pair_distance_matrix
    row_norms = 0.0d0

    !$OMP PARALLEL DO PRIVATE(pair_norm) REDUCTION(+:row_norms)
    do i = 1, natoms
        pair_norm = 0.5d0 * atomic_charges(i) ** 2.4d0
        row_norms(i) = row_norms(i) + pair_norm * pair_norm
        pair_distance_matrix(i, i) = pair_norm
    enddo
    !$OMP END PARALLEL DO

    !$OMP PARALLEL DO PRIVATE(pair_norm) REDUCTION(+:row_norms)
    do i = 1, natoms
        do j = i+1, natoms
            pair_norm = atomic_charges(i) * atomic_charges(j) &
                & / sqrt(sum((coordinates(j,:) - coordinates(i,:))**2))

            pair_distance_matrix(i, j) = pair_norm
            pair_distance_matrix(j, i) = pair_norm
            pair_norm = pair_norm * pair_norm
            row_norms(j) = row_norms(j) + pair_norm
            row_norms(i) = row_norms(i) + pair_norm
        enddo
    enddo
    !$OMP END PARALLEL DO

    !Generate sorted list of atom ids by row_norms - not really (easily) parallelizable
    do i = 1, natoms
        j = minloc(row_norms, dim=1)
        sorted_atoms(natoms - i + 1) = j
        row_norms(j) = huge_double
    enddo

    ! Fill coulomb matrix according to sorted row-2-norms
    cm = 0.0d0
    !$OMP PARALLEL DO PRIVATE(idx, i, j)
    do m = 1, natoms
        i = sorted_atoms(m)
        idx = (m*m+m)/2 - m
        do n = 1, m
            j = sorted_atoms(n)
            cm(idx+n) = pair_distance_matrix(i, j)
        enddo
    enddo
    !$OMP END PARALLEL DO

    ! Clean up
    deallocate(pair_distance_matrix)
    deallocate(row_norms)
    deallocate(sorted_atoms)
end subroutine fgenerate_coulomb_matrix

subroutine fgenerate_unsorted_coulomb_matrix(atomic_charges, coordinates, nmax, cm)

    implicit none

    double precision, dimension(:), intent(in) :: atomic_charges
    double precision, dimension(:,:), intent(in) :: coordinates

    integer, intent(in) :: nmax

    double precision, dimension(((nmax + 1) * nmax) / 2), intent(out):: cm

    double precision :: pair_norm

    double precision, allocatable, dimension(:,:) :: pair_distance_matrix

    integer :: i, j, m, n, idx
    integer :: natoms

    if (size(coordinates, dim=1) /= size(atomic_charges, dim=1)) then
        write(*,*) "ERROR: Coulomb matrix generation"
        write(*,*) size(coordinates, dim=1), "coordinates, but", &
            & size(atomic_charges, dim=1), "atom_types!"
        stop
    else
        natoms = size(atomic_charges, dim=1)
    endif

    ! Allocate temporary
    allocate(pair_distance_matrix(natoms,natoms))

    !$OMP PARALLEL DO PRIVATE(pair_norm)
    do i = 1, natoms
        pair_norm = 0.5d0 * atomic_charges(i) ** 2.4d0
        pair_distance_matrix(i, i) = pair_norm
    enddo
    !$OMP END PARALLEL DO

    !$OMP PARALLEL DO PRIVATE(pair_norm)
    do i = 1, natoms
        do j = i+1, natoms
            pair_norm = atomic_charges(i) * atomic_charges(j) &
                & / sqrt(sum((coordinates(j,:) - coordinates(i,:))**2))

            pair_distance_matrix(i, j) = pair_norm
            pair_distance_matrix(j, i) = pair_norm
        enddo
    enddo
    !$OMP END PARALLEL DO


    cm = 0.0d0
    !$OMP PARALLEL DO PRIVATE(idx)
    do m = 1, natoms
        idx = (m*m+m)/2 - m
        do n = 1, m
            cm(idx+n) = pair_distance_matrix(m, n)
        enddo
    enddo
    !$OMP END PARALLEL DO

    ! Clean up
    deallocate(pair_distance_matrix)

end subroutine fgenerate_unsorted_coulomb_matrix

subroutine fgenerate_local_coulomb_matrix(central_atom_indices, central_natoms, &
        & atomic_charges, coordinates, natoms, nmax, cent_cutoff, cent_decay, &
        & int_cutoff, int_decay, cm)

    implicit none

    integer, intent(in) :: central_natoms
    integer, dimension(:), intent(in) :: central_atom_indices
    double precision, dimension(:), intent(in) :: atomic_charges
    double precision, dimension(:,:), intent(in) :: coordinates
    integer,intent(in) :: natoms
    integer, intent(in) :: nmax
    double precision, intent(inout) :: cent_cutoff, cent_decay, int_cutoff, int_decay

    double precision, dimension(central_natoms, ((nmax + 1) * nmax) / 2), intent(out):: cm

    integer :: idx

    double precision, allocatable, dimension(:, :) :: row_norms
    double precision :: pair_norm
    double precision :: prefactor
    double precision :: norm
    double precision :: huge_double

    integer, allocatable, dimension(:, :) :: sorted_atoms_all
    integer, allocatable, dimension(:) :: cutoff_count

    double precision, allocatable, dimension(:, :, :) :: pair_distance_matrix
    double precision, allocatable, dimension(:, :) :: distance_matrix

    integer i, j, m, n, k, l

    double precision, parameter :: pi = 4.0d0 * atan(1.0d0)


    if (size(coordinates, dim=1) /= size(atomic_charges, dim=1)) then
        write(*,*) "ERROR: Coulomb matrix generation"
        write(*,*) size(coordinates, dim=1), "coordinates, but", &
            & size(atomic_charges, dim=1), "atom_types!"
        stop
    endif

    ! Allocate temporary
    allocate(distance_matrix(natoms,natoms))
    allocate(cutoff_count(natoms))

    huge_double = huge(distance_matrix(1,1))

    if (cent_cutoff < 0) then
        cent_cutoff = huge_double
    endif

    if ((int_cutoff < 0) .OR. (int_cutoff > 2 * cent_cutoff)) then
        int_cutoff = 2 * cent_cutoff
    endif

    if (cent_decay < 0) then
        cent_decay = 0.0d0
    else if (cent_decay > cent_cutoff) then
        cent_decay = cent_cutoff
    endif

    if (int_decay < 0) then
        int_decay = 0.0d0
    else if (int_decay > int_cutoff) then
        int_decay = int_cutoff
    endif


    cutoff_count = 1

    !$OMP PARALLEL DO PRIVATE(norm) REDUCTION(+:cutoff_count)
    do i = 1, natoms
        distance_matrix(i, i) = 0.0d0
        do j = i+1, natoms
            norm = sqrt(sum((coordinates(j,:) - coordinates(i,:))**2))
            distance_matrix(i, j) = norm
            distance_matrix(j, i) = norm
            if (norm < cent_cutoff) then
                cutoff_count(i) = cutoff_count(i) + 1
                cutoff_count(j) = cutoff_count(j) + 1
            endif
        enddo
    enddo
    !$OMP END PARALLEL DO

    do i = 1, central_natoms
        j = central_atom_indices(i)
        if (cutoff_count(j) > nmax) then
            write(*,*) "ERROR: Coulomb matrix generation"
            write(*,*) nmax, "size set, but", &
                & cutoff_count(j), "size needed!"
            stop
        endif
    enddo

    ! Allocate temporary
    allocate(pair_distance_matrix(natoms, natoms, central_natoms))
    allocate(row_norms(natoms, central_natoms))

    pair_distance_matrix = 0.0d0
    row_norms = 0.0d0


    !$OMP PARALLEL DO PRIVATE(pair_norm, prefactor, k) REDUCTION(+:row_norms) COLLAPSE(2)
    do i = 1, natoms
        do l = 1, central_natoms
            k = central_atom_indices(l)
            ! self interaction
            if (distance_matrix(i,k) > cent_cutoff) cycle

            prefactor = 1.0d0
            if (distance_matrix(i,k) > cent_cutoff - cent_decay) then
                prefactor = 0.5d0 * (cos(pi &
                    & * (distance_matrix(i,k) - cent_cutoff + cent_decay) / cent_decay) + 1)
            endif

            pair_norm = prefactor * prefactor * 0.5d0 * atomic_charges(i) ** 2.4d0
            pair_distance_matrix(i,i,l) = pair_norm
            row_norms(i,l) = row_norms(i,l) + pair_norm * pair_norm

            do j = i+1, natoms
                if (distance_matrix(j,k) > cent_cutoff) cycle

                if (distance_matrix(i,j) > int_cutoff) cycle

                pair_norm = prefactor * atomic_charges(i) * atomic_charges(j) &
                    & / distance_matrix(j, i)

                if (distance_matrix(i,j) > int_cutoff - int_decay) then
                    pair_norm = pair_norm * 0.5d0 * (cos(pi &
                        & * (distance_matrix(i,j) - int_cutoff + int_decay) / int_decay) + 1)
                endif


                if (distance_matrix(j,k) > cent_cutoff - cent_decay) then
                    pair_norm = pair_norm * 0.5d0 * (cos(pi &
                        & * (distance_matrix(j,k) - cent_cutoff + cent_decay) / cent_decay) + 1)
                endif

                pair_distance_matrix(i, j, l) = pair_norm
                pair_distance_matrix(j, i, l) = pair_norm
                pair_norm = pair_norm * pair_norm
                row_norms(i,l) = row_norms(i,l) + pair_norm
                row_norms(j,l) = row_norms(j,l) + pair_norm
            enddo
        enddo
    enddo
    !$OMP END PARALLEL DO

    ! Allocate temporary
    allocate(sorted_atoms_all(natoms, central_natoms))

    !$OMP PARALLEL DO PRIVATE(k)
        do l = 1, central_natoms
            k = central_atom_indices(l)
            row_norms(k,l) = huge_double
        enddo
    !$OMP END PARALLEL DO

    !$OMP PARALLEL DO PRIVATE(j,k)
        do l = 1, central_natoms
            k = central_atom_indices(l)
            !$OMP CRITICAL
                do i = 1, cutoff_count(k)
                    j = maxloc(row_norms(:,l), dim=1)
                    sorted_atoms_all(i, l) = j
                    row_norms(j,l) = 0.0d0
                enddo
            !$OMP END CRITICAL
        enddo
    !$OMP END PARALLEL DO

    ! Clean up
    deallocate(row_norms)



    ! Fill coulomb matrix according to sorted row-2-norms
    cm = 0.0d0

    !$OMP PARALLEL DO PRIVATE(i, j, k, idx)
        do l = 1, central_natoms
            k = central_atom_indices(l)
            do m = 1, cutoff_count(k)
                i = sorted_atoms_all(m, l)
                idx = (m*m+m)/2 - m
                do n = 1, m
                    j = sorted_atoms_all(n, l)
                    cm(l, idx+n) = pair_distance_matrix(i,j,l)
                enddo
            enddo
        enddo
    !$OMP END PARALLEL DO


    ! Clean up
    deallocate(sorted_atoms_all)
    deallocate(pair_distance_matrix)

end subroutine fgenerate_local_coulomb_matrix

subroutine fgenerate_atomic_coulomb_matrix(central_atom_indices, central_natoms, atomic_charges, &
        & coordinates, natoms, nmax, cent_cutoff, cent_decay, int_cutoff, int_decay, cm)

    implicit none

    integer, dimension(:), intent(in) :: central_atom_indices
    integer, intent(in) :: central_natoms
    double precision, dimension(:), intent(in) :: atomic_charges
    double precision, dimension(:,:), intent(in) :: coordinates
    integer,intent(in) :: natoms
    integer, intent(in) :: nmax
    double precision, intent(inout) :: cent_cutoff, cent_decay, int_cutoff, int_decay

    double precision, dimension(central_natoms, ((nmax + 1) * nmax) / 2), intent(out):: cm

    integer :: idx

    double precision :: pair_norm
    double precision :: prefactor
    double precision :: norm
    double precision :: huge_double

    integer, allocatable, dimension(:, :) :: sorted_atoms_all
    integer, allocatable, dimension(:) :: cutoff_count

    double precision, allocatable, dimension(:, :) :: pair_distance_matrix
    double precision, allocatable, dimension(:, :) :: distance_matrix
    double precision, allocatable, dimension(:, :) :: distance_matrix_tmp

    integer i, j, m, n, k, l

    double precision, parameter :: pi = 4.0d0 * atan(1.0d0)

    if (size(coordinates, dim=1) /= size(atomic_charges, dim=1)) then
        write(*,*) "ERROR: Coulomb matrix generation"
        write(*,*) size(coordinates, dim=1), "coordinates, but", &
            & size(atomic_charges, dim=1), "atom_types!"
        stop
    endif

    ! Allocate temporary
    allocate(distance_matrix(natoms,natoms))
    allocate(cutoff_count(natoms))

    huge_double = huge(distance_matrix(1,1))

    if (cent_cutoff < 0) then
        cent_cutoff = huge_double
    endif

    if ((int_cutoff < 0) .OR. (int_cutoff > 2 * cent_cutoff)) then
        int_cutoff = 2 * cent_cutoff
    endif

    if (cent_decay < 0) then
        cent_decay = 0.0d0
    else if (cent_decay > cent_cutoff) then
        cent_decay = cent_cutoff
    endif

    if (int_decay < 0) then
        int_decay = 0.0d0
    else if (int_decay > int_cutoff) then
        int_decay = int_cutoff
    endif


    cutoff_count = 1

    !$OMP PARALLEL DO PRIVATE(norm) REDUCTION(+:cutoff_count)
    do i = 1, natoms
        distance_matrix(i, i) = 0.0d0
        do j = i+1, natoms
            norm = sqrt(sum((coordinates(j,:) - coordinates(i,:))**2))
            distance_matrix(i, j) = norm
            distance_matrix(j, i) = norm
            if (norm < cent_cutoff) then
                cutoff_count(i) = cutoff_count(i) + 1
                cutoff_count(j) = cutoff_count(j) + 1
            endif
        enddo
    enddo
    !$OMP END PARALLEL DO

    do i = 1, central_natoms
        k = central_atom_indices(i)
        if (cutoff_count(k) > nmax) then
            write(*,*) "ERROR: Coulomb matrix generation"
            write(*,*) nmax, "size set, but", &
                & cutoff_count(k), "size needed!"
            stop
        endif
    enddo

    ! Allocate temporary
    allocate(pair_distance_matrix(natoms, natoms))

    pair_distance_matrix = 0.0d0

    !$OMP PARALLEL DO PRIVATE(pair_norm)
        do i = 1, natoms
            pair_distance_matrix(i, i) = 0.5d0 * atomic_charges(i) ** 2.4d0
            do j = i+1, natoms
                if (distance_matrix(j,i) > int_cutoff) cycle
                pair_norm = atomic_charges(i) * atomic_charges(j) &
                    & / distance_matrix(j, i)
                if (distance_matrix(j,i) > int_cutoff - int_decay) then
                    pair_norm = pair_norm * 0.5d0 * (cos(pi &
                        & * (distance_matrix(j,i) - int_cutoff + int_decay) / int_decay) + 1)
                endif

                pair_distance_matrix(i, j) = pair_norm
                pair_distance_matrix(j, i) = pair_norm
            enddo
        enddo
    !$OMP END PARALLEL DO

    ! Allocate temporary
    allocate(sorted_atoms_all(natoms, natoms))
    allocate(distance_matrix_tmp(natoms, natoms))

    distance_matrix_tmp = distance_matrix
    !Generate sorted list of atom ids by distance matrix
    !$OMP PARALLEL DO PRIVATE(j, k)
    do l = 1, central_natoms
        k = central_atom_indices(l)
        !$OMP CRITICAL
            do i = 1, cutoff_count(k)
                j = minloc(distance_matrix_tmp(:,k), dim=1)
                sorted_atoms_all(i, l) = j
                distance_matrix_tmp(j, k) = huge_double
            enddo
        !$OMP END CRITICAL
    enddo
    !$OMP END PARALLEL DO

    ! Clean up
    deallocate(distance_matrix_tmp)

    ! Fill coulomb matrix according to sorted distances
    cm = 0.0d0

    pair_norm = 0.0d0

    !$OMP PARALLEL DO PRIVATE(i, prefactor, idx, j, pair_norm, k)
    do l = 1, central_natoms
        k = central_atom_indices(l)
        do m = 1, cutoff_count(k)
            i = sorted_atoms_all(m, l)

                if (distance_matrix(i,k) > cent_cutoff) cycle
                prefactor = 1.0d0
                if (distance_matrix(i,k) > cent_cutoff - cent_decay) then
                    prefactor = 0.5d0 * (cos(pi &
                        & * (distance_matrix(i,k) - cent_cutoff + cent_decay) &
                        & / cent_decay) + 1.0d0)
                endif

            idx = (m*m+m)/2 - m
            do n = 1, m
                j = sorted_atoms_all(n, l)

                pair_norm = prefactor * pair_distance_matrix(i, j)
                if (distance_matrix(j,k) > cent_cutoff - cent_decay) then
                    pair_norm = pair_norm * 0.5d0 * (cos(pi &
                        & * (distance_matrix(j,k) - cent_cutoff + cent_decay) &
                        & / cent_decay) + 1)
                endif
                cm(l, idx+n) = pair_norm
            enddo
        enddo
    enddo

    ! Clean up
    deallocate(distance_matrix)
    deallocate(pair_distance_matrix)
    deallocate(sorted_atoms_all)
    deallocate(cutoff_count)

end subroutine fgenerate_atomic_coulomb_matrix

subroutine fgenerate_eigenvalue_coulomb_matrix(atomic_charges, coordinates, nmax, sorted_eigenvalues)

    implicit none

    double precision, dimension(:), intent(in) :: atomic_charges
    double precision, dimension(:,:), intent(in) :: coordinates

    integer, intent(in) :: nmax

    double precision, dimension(nmax), intent(out) :: sorted_eigenvalues

    double precision :: pair_norm
    double precision :: huge_double

    double precision, allocatable, dimension(:,:) :: pair_distance_matrix

    double precision, allocatable, dimension(:) :: work
    double precision, allocatable, dimension(:) :: eigenvalues

    integer :: i, j, info, lwork
    integer :: natoms

    if (size(coordinates, dim=1) /= size(atomic_charges, dim=1)) then
        write(*,*) "ERROR: Coulomb matrix generation"
        write(*,*) size(coordinates, dim=1), "coordinates, but", &
            & size(atomic_charges, dim=1), "atom_types!"
        stop
    else
        natoms = size(atomic_charges, dim=1)
    endif

    ! Allocate temporary
    allocate(pair_distance_matrix(nmax,nmax))

    huge_double = huge(pair_distance_matrix(1,1))

    pair_distance_matrix(:,:) = 0.0d0

    !$OMP PARALLEL DO PRIVATE(pair_norm)
    do i = 1, natoms
        pair_norm = 0.5d0 * atomic_charges(i) ** 2.4d0
        pair_distance_matrix(i, i) = pair_norm
    enddo
    !$OMP END PARALLEL DO

    !$OMP PARALLEL DO PRIVATE(pair_norm)
    do i = 1, natoms
        do j = i+1, natoms
            pair_norm = atomic_charges(i) * atomic_charges(j) &
                & / sqrt(sum((coordinates(j,:) - coordinates(i,:))**2))

            pair_distance_matrix(i, j) = pair_norm
        enddo
    enddo
    !$OMP END PARALLEL DO


    lwork = 4 * nmax
    ! Allocate temporary
    allocate(work(lwork))
    allocate(eigenvalues(nmax))
    call dsyev("N", "U", nmax, pair_distance_matrix, nmax, eigenvalues, work, lwork, info)
    if (info > 0) then
        write (*,*) "WARNING: Eigenvalue routine DSYEV() exited with error code:", info
    endif

    ! Clean up
    deallocate(work)
    deallocate(pair_distance_matrix)

    !sort
    do i = 1, nmax
        j = minloc(eigenvalues, dim=1)
        sorted_eigenvalues(nmax - i + 1) = eigenvalues(j)
        eigenvalues(j) = huge_double
    enddo

    ! Clean up
    deallocate(eigenvalues)


end subroutine fgenerate_eigenvalue_coulomb_matrix

subroutine fgenerate_bob(atomic_charges, coordinates, nuclear_charges, id, &
    & nmax, ncm, cm)

    use representations, only: get_indices
    implicit none

    double precision, dimension(:), intent(in) :: atomic_charges
    double precision, dimension(:,:), intent(in) :: coordinates
    integer, dimension(:), intent(in) :: nuclear_charges
    integer, dimension(:), intent(in) :: id
    integer, dimension(:), intent(in) :: nmax
    integer, intent(in) :: ncm

    double precision, dimension(ncm), intent(out):: cm

    integer :: n, i, j, k, l, idx1, idx2, nid, nbag
    integer :: natoms, natoms1, natoms2, type1, type2

    integer, allocatable, dimension(:) :: type1_indices
    integer, allocatable, dimension(:) :: type2_indices
    integer, allocatable, dimension(:,:) :: start_indices


    double precision :: pair_norm
    double precision :: huge_double

    double precision, allocatable, dimension(:) :: bag
    double precision, allocatable, dimension(:,:) :: pair_distance_matrix


    if (size(coordinates, dim=1) /= size(atomic_charges, dim=1)) then
        write(*,*) "ERROR: Bag of Bonds generation"
        write(*,*) size(coordinates, dim=1), "coordinates, but", &
            & size(atomic_charges, dim=1), "atom_types!"
        stop
    else if (size(coordinates, dim=1) /= size(nuclear_charges, dim=1)) then
        write(*,*) "ERROR: Coulomb matrix generation"
        write(*,*) size(coordinates, dim=1), "coordinates, but", &
            & size(nuclear_charges, dim=1), "atom_types!"
        stop
    else
        natoms = size(atomic_charges, dim=1)
    endif

    if (size(id, dim=1) /= size(nmax, dim=1)) then
        write(*,*) "ERROR: Bag of Bonds generation"
        write(*,*) size(id, dim=1), "unique atom types, but", &
            & size(nmax, dim=1), "max size!"
        stop
    else
        nid = size(id, dim=1)
    endif

    n = 0
    !$OMP PARALLEL DO REDUCTION(+:n)
        do i = 1, nid
            n = n + nmax(i) * (1 + nmax(i))
            do j = 1, i - 1
                n = n + 2 * nmax(i) * nmax(j)
            enddo
        enddo
    !$OMP END PARALLEL DO

    if (n /= 2*ncm) then
        write(*,*) "ERROR: Bag of Bonds generation"
        write(*,*) "Inferred vector size", n, "but given size", ncm
        stop
    endif

    ! Allocate temporary
    allocate(pair_distance_matrix(natoms,natoms))
    huge_double = huge(pair_distance_matrix(1,1))


    !$OMP PARALLEL DO PRIVATE(pair_norm)
    do i = 1, natoms
        do j = i+1, natoms
            pair_norm = atomic_charges(i) * atomic_charges(j) &
                & / sqrt(sum((coordinates(j,:) - coordinates(i,:))**2))

            pair_distance_matrix(i, j) = pair_norm
            pair_distance_matrix(j, i) = pair_norm
        enddo
    enddo
    !$OMP END PARALLEL DO

    ! Allocate temporary
    ! Too large but easier
    allocate(type1_indices(maxval(nmax, dim=1)))
    allocate(type2_indices(maxval(nmax, dim=1)))

    ! Get max bag size
    nbag = 0
    do i = 1, nid
        nbag = max(nbag, (nmax(i) * (nmax(i) - 1))/2)
        do j = 1, i - 1
            nbag = max(nbag, nmax(i) * nmax(j))
        enddo
    enddo

    ! Allocate temporary
    ! Too large but easier
    allocate(bag(nbag))
    allocate(start_indices(nid,nid))

    ! get start indices
    do i = 1, nid
        if (i == 1) then
            start_indices(1,1) = 0
        else
            start_indices(i,i) = start_indices(i-1,nid) + nmax(i-1)*nmax(nid)
        endif

        if (nid > i) then
            start_indices(i, i+1) = start_indices(i,i) + (nmax(i) * (nmax(i) + 1)) / 2
            do j = i + 2, nid
                start_indices(i,j) = start_indices(i,j-1) + nmax(i)*nmax(j-1)
            enddo
        endif
    enddo

    cm = 0.0d0

    !$OMP PARALLEL DO PRIVATE(type1, type1_indices, l, &
    !$OMP& bag, natoms1, idx1, idx2, k, nbag, type2, natoms2, type2_indices) &
    !$OMP& SCHEDULE(dynamic)
    do i = 1, nid
        type1 = id(i)
        natoms1 = 0

        call get_indices(natoms, nuclear_charges, type1, natoms1, type1_indices)

        do j = 1, natoms1
            idx1 = type1_indices(j)
            cm(start_indices(i,i) + j) = 0.5d0 * atomic_charges(idx1) ** 2.4d0
            k = (j * j - 3 * j) / 2
            do l = 1, j - 1
                idx2 = type1_indices(l)
                bag(k + l + 1) = pair_distance_matrix(idx1, idx2)
            enddo
        enddo

        nbag = (natoms1 * natoms1 - natoms1) / 2
        ! sort
        do j = 1, nbag
            k = minloc(bag(:nbag), dim=1)
            cm(start_indices(i,i) + nmax(i) + nbag - j + 1) = bag(k)
            bag(k) = huge_double
        enddo

        do j = i + 1, nid
            type2 = id(j)
            natoms2 = 0

            call get_indices(natoms, nuclear_charges, type2, natoms2, type2_indices)

            do k = 1, natoms1
                idx1 = type1_indices(k)
                do l = 1, natoms2
                    idx2 = type2_indices(l)
                    bag(natoms2 * (k - 1) + l) = pair_distance_matrix(idx1, idx2)
                enddo
            enddo

            ! sort
            nbag = natoms1 * natoms2
            do k = 1, nbag
                l = minloc(bag(:nbag), dim=1)
                cm(start_indices(i,j) + nbag - k + 1) = bag(l)
                bag(l) = huge_double
            enddo

        enddo
    enddo
    !$OMP END PARALLEL DO

    ! Clean up
    deallocate(pair_distance_matrix)
    deallocate(bag)
    deallocate(type1_indices)
    deallocate(type2_indices)
    deallocate(start_indices)

end subroutine fgenerate_bob

subroutine fgenerate_local_bob(atomic_charges, coordinates, nuclear_charges, id, &
        & nmax, natoms, cent_cutoff, cent_decay, int_cutoff, int_decay, ncm, cm)

    use omp_lib

    implicit none

    double precision, dimension(:), intent(in) :: atomic_charges
    double precision, dimension(:,:), intent(in) :: coordinates
    integer, dimension(:), intent(in) :: nuclear_charges
    integer, dimension(:), intent(in) :: id
    integer, dimension(:), intent(in) :: nmax
    integer, intent(in) :: natoms
    double precision, intent(inout) :: cent_cutoff, cent_decay, int_cutoff, int_decay
    integer, intent(in) :: ncm
    double precision, dimension(natoms, ncm), intent(out):: cm

    double precision :: pair_norm
    double precision :: norm
    double precision :: decay_dist
    double precision :: huge_double
    double precision :: prefactor

    integer, allocatable, dimension(:, :, :) :: start_indices
    integer, allocatable, dimension(:) :: index_types
    integer, allocatable, dimension(:, :) :: n_pair_indices
    integer, allocatable, dimension(:, :, :) :: pair_indices
    integer, allocatable, dimension(:, :, :, :) :: n_all_pair_indices
    integer, allocatable, dimension(:, :, :, :, :) :: all_pair_indices

    logical, allocatable, dimension(:, :) :: mask

    integer(kind = OMP_lock_kind), allocatable, dimension(:,:) :: lock

    double precision, allocatable, dimension(:) :: bag
    double precision, allocatable, dimension(:, :) :: distance_matrix
    double precision, allocatable, dimension(:, :) :: cent_decay_matrix
    double precision, allocatable, dimension(:, :) :: int_decay_matrix

    integer :: i, j, k, l, m, n, nid, type0, type1, type2, nbag, max_idx

    double precision, parameter :: pi = 4.0d0 * atan(1.0d0)

    write(*,*) "start"

    ! size checks
    if (size(coordinates, dim=1) /= size(atomic_charges, dim=1)) then
        write(*,*) "ERROR: Coulomb matrix generation"
        write(*,*) size(coordinates, dim=1), "coordinates, but", &
            & size(atomic_charges, dim=1), "atom_types!"
        stop
    else if (size(coordinates, dim=1) /= size(nuclear_charges, dim=1)) then
        write(*,*) "ERROR: Coulomb matrix generation"
        write(*,*) size(coordinates, dim=1), "coordinates, but", &
            & size(nuclear_charges, dim=1), "atom_types!"
        stop
    endif

    if (size(id, dim=1) /= size(nmax, dim=1)) then
        write(*,*) "ERROR: Bag of Bonds generation"
        write(*,*) size(id, dim=1), "unique atom types, but", &
            & size(nmax, dim=1), "max size!"
        stop
    else
        nid = size(id, dim=1)
    endif

    n = 1
    do i = 1, nid
        n = n + (nmax(i) * (nmax(i) - 1)) / 2
        n = n + nmax(i) * 2
        do j = i + 1, nid
            n = n + nmax(i) * nmax(j)
        enddo
    enddo

    if (n /= ncm) then
        write(*,*) "ERROR: Bag of Bonds generation"
        write(*,*) "Inferred vector size", n, "but given size", ncm
        stop
    endif


    ! Fix cutoffs and decays outside legal range
    if (cent_cutoff < 0) then
        cent_cutoff = huge_double
    endif

    if ((int_cutoff < 0) .OR. (int_cutoff > 2 * cent_cutoff)) then
        int_cutoff = 2 * cent_cutoff
    endif

    if (cent_decay < 0) then
        cent_decay = 0.0d0
    else if (cent_decay > cent_cutoff) then
        cent_decay = cent_cutoff
    endif

    if (int_decay < 0) then
        int_decay = 0.0d0
    else if (int_decay > int_cutoff) then
        int_decay = int_cutoff
    endif

    ! Allocate temporary
    allocate(lock(natoms, nid))

    !$OMP PARALLEL DO
        do i = 1, natoms
            do j = 1, nid
                call OMP_init_lock(lock(i,j))
            enddo
        enddo
    !$OMP END PARALLEL DO


    ! Allocate temporary
    allocate(index_types(natoms))

    index_types = 1

    ! Get element types
    if (nid > 2) then
        !$OMP PARALLEL DO PRIVATE(type1)
        do i = 1, natoms
            type1 = nuclear_charges(i)
            do j = 2, nid
                if (type1 == id(j)) then
                    index_types(i) = j
                endif
            enddo
        enddo
        !$OMP END PARALLEL DO
    endif
    write(*,*) "start2"

    ! Allocate temporary
    allocate(distance_matrix(natoms,natoms))
    allocate(n_pair_indices(natoms, nid))
    allocate(pair_indices(natoms, nid, natoms))
    allocate(mask(natoms, natoms))
    allocate(cent_decay_matrix(natoms,natoms))
    allocate(int_decay_matrix(natoms,natoms))

    huge_double = huge(distance_matrix(1,1))
    n_pair_indices = 0
    mask = .false.
    cent_decay_matrix = 1.0d0
    int_decay_matrix = 1.0d0

    ! Create convenience arrays
    !$OMP PARALLEL DO PRIVATE(norm, type1, type2, prefactor, decay_dist) &
    !$OMP& REDUCTION(*:cent_decay_matrix,int_decay_matrix) SCHEDULE(guided)
        do i = 1, natoms
            distance_matrix(i, i) = 0.0d0
            type1 = index_types(i)
            do j = 1, i-1
                type2 = index_types(j)
                ! Get distances
                norm = sqrt(sum((coordinates(i,:) - coordinates(j,:))**2))
                distance_matrix(i, j) = norm
                distance_matrix(j, i) = norm
                if (norm < cent_cutoff) then
                    ! Store indices that yield non-zero contributions
                    call OMP_set_lock(lock(i, type2))
                    n_pair_indices(i, type2) = n_pair_indices(i, type2) + 1
                    pair_indices(i, type2, n_pair_indices(i, type2)) = j
                    call OMP_unset_lock(lock(i, type2))

                    call OMP_set_lock(lock(j, type1))
                    n_pair_indices(j, type1) = n_pair_indices(j, type1) + 1
                    pair_indices(j, type1, n_pair_indices(j, type1)) = i
                    call OMP_unset_lock(lock(j, type1))

                    ! Calculate decay factors
                    decay_dist = norm - cent_cutoff + cent_decay
                    if (decay_dist > 0) then
                        prefactor = 0.5d0 * &
                            & (cos(pi * decay_dist / cent_decay) + 1)

                        cent_decay_matrix(i,j) = cent_decay_matrix(i,j) * prefactor
                        cent_decay_matrix(j,i) = cent_decay_matrix(j,i) * prefactor

                    endif
                else
                    cycle
                endif

                if (norm > int_cutoff) then
                    mask(i,j) = .true.
                    mask(j,i) = .true.
                else
                    decay_dist = norm - int_cutoff + int_decay
                    if (decay_dist > 0) then
                        prefactor = 0.5d0 * &
                            & (cos(pi * decay_dist / int_decay) + 1)

                        int_decay_matrix(i,j) = int_decay_matrix(i,j) * prefactor
                        int_decay_matrix(j,i) = int_decay_matrix(j,i) * prefactor
                    endif
                endif
            enddo
        enddo
    !$OMP END PARALLEL DO

    ! Clean up
    deallocate(lock)

    ! Check given array sizes
    do j = 1, nid
        n = maxval(n_pair_indices(:,j), dim=1)
        if (n > nmax(j)) then
            write(*,*) "ERROR: Bag of Bonds generation"
            write(*,*) nmax(j), "size set for atom with nuclear charge,", &
                & id(j), "but", n, "size needed!"
        endif
    enddo

    write(*,*) "start3"

    ! Allocate temporary
    max_idx = 1 ! to store k index
    do j = 1, nid
        max_idx = max(max_idx,maxval(n_pair_indices(:,j), dim=1))
    enddo
    allocate(n_all_pair_indices(natoms, nid, nid, max_idx))
    allocate(all_pair_indices(natoms,nid,nid,max_idx, max_idx-1))

    n_all_pair_indices = 0

    !$OMP PARALLEL DO PRIVATE(l, m, n, type0, type1, type2)
        do k = 1, natoms
            type0 = index_types(k)
            ! Only fill the indices needed for the bags.
            do type1 = 1, nid
                do i = 1, n_pair_indices(k, type1)
                    l = pair_indices(k, type1, i)

                    ! bags with interactions between atoms of same atom type
                    do j = i+1, n_pair_indices(k, type1)
                        m = pair_indices(k, type1, j)
                        if (mask(l,m)) cycle

                        n = n_all_pair_indices(k, type1, type1, i) + 1
                        n_all_pair_indices(k, type1, type1, i) = n
                        all_pair_indices(k, type1, type1, i, n) = m
                    enddo

                    ! bags with interactions between atoms of different atom type
                    do type2 = type1+1, nid
                        do j = 1, n_pair_indices(k, type2)
                            m = pair_indices(k, type2, j)
                            if (mask(l,m)) cycle
                            if (m == k) cycle

                            n = n_all_pair_indices(k, type1, type2, i) + 1
                            n_all_pair_indices(k, type1, type2, i) = n
                            all_pair_indices(k, type1, type2, i, n) = m
                        enddo
                    enddo

                    ! Indices for XA bag
                    ! Alternatively, don't include int_cutoff (mask)
                    if (mask(l,k)) cycle
                    n = n_all_pair_indices(k, type0, type1, max_idx) + 1
                    n_all_pair_indices(k, type0, type1, max_idx) = n
                    all_pair_indices(k, type0, type1, max_idx, n) = l
                enddo
            enddo
        enddo
    !$OMP END PARALLEL DO

    ! Clean up
    deallocate(mask)

    ! Allocate temporary
    allocate(start_indices(nid,nid,3))

    ! get start indices
    ! bags with atom types A,B,C with k of type X in (A,B,C):
    ! (X, A, B, C, XA, XB, XC, AA, AB, AC, BB, BC, CC)
    ! A
    do i = 1, nid
        if (i == 1) then
            start_indices(1,1,1) = 1
        else
            start_indices(i,1,1) = start_indices(i-1,1,1) + nmax(i-1)
        endif
    enddo
    ! XA
    do i = 1, nid
        if (i == 1) then
            start_indices(1,1,2) = start_indices(nid,1,1) + nmax(nid)
        else
            start_indices(i,1,2) = start_indices(i-1,1,2) + nmax(i-1)
        endif
    enddo
    write(*,*) "start4"

    ! AB (including AA)
    do i = 1, nid
        do j = i, nid
            if (j == 1) then
                start_indices(1,1,3) = start_indices(nid,1,2) + (nmax(nid)* (nmax(nid) - 1)) / 2
            else if (j == i) then
                start_indices(i,i,3) = start_indices(i-1,nid,3) + nmax(i-1)*nmax(nid)
            else
                start_indices(i,j,3) = start_indices(i,j-1,3) + nmax(i)*nmax(j-1)
            endif
        enddo
    enddo

    ! Get max bag size
    nbag = 1
    do i = 1, nid
        nbag = max(nbag, nmax(i))
        nbag = max(nbag, (nmax(i)*(nmax(i)-1))/2)
        do j = i + 1, nid
            nbag = max(nbag, nmax(i)*nmax(j))
        enddo
    enddo

    ! Allocate temporary
    allocate(bag(nbag))

    ! Construct representation
    cm = 0.0d0

    write(*,*) "start5"
        do k = 1, natoms
            write(*,*) "k", k
            type0 = index_types(k)

            ! X bag
            cm(k,1) = 0.5d0 * atomic_charges(k) ** 2.4d0

            write(*,*) "start A"
            !A bag (self-interactions)
            do type1 = 1, nid
                nbag = n_pair_indices(k, type1)
                do i = 1, nbag
                    l = pair_indices(k, type1, i)

                    norm = distance_matrix(l, k)
                    pair_norm = 0.5d0 * atomic_charges(l) ** 2.4d0 &
                        & * cent_decay_matrix(k,i) ** 2
                    bag(i) = pair_norm
                enddo

                ! sort
                n = start_indices(type1, 1, 1)
                do i = 1, nbag
                    l = maxloc(bag(:nbag), dim=1)
                    cm(k, n + i) = bag(l)
                    bag(l) = -huge_double
                enddo
                ! end sort
            enddo
            ! end A bag

            write(*,*) "start XA"
                        write(*,*) n_all_pair_indices(5,2,2,1)
            ! XA bag
            do type1 = 1, nid
                nbag = n_all_pair_indices(k, type0, type1, max_idx)

                do i = 1, nbag
                    l = all_pair_indices(k, type0, type1, max_idx, i)

                    norm = distance_matrix(l, k)

                    ! Alternatively, don't include int_decay
                    pair_norm = atomic_charges(k) * atomic_charges(l) / norm &
                        & * cent_decay_matrix(k,l) &
                        & * int_decay_matrix(k,l)

                    bag(i) = pair_norm
                enddo

                ! sort
                n = start_indices(type1, 1, 2)
                do i = 1, nbag
                    l = maxloc(bag(:nbag), dim=1)
                    cm(k, n + i) = bag(l)
                    bag(l) = -huge_double
                enddo
                ! end sort
            enddo

            write(*,*) "start AB"
                        write(*,*) n_all_pair_indices(5,2,2,1)
            ! AB bag including AA
            do type1 = 1, nid
                write(*,*) "type1", type1
                do type2 = type1, nid
                    write(*,*) "type2", type2
                    nbag = 0
                    do i = 1, n_pair_indices(k, type1)
                        write(*,*) "i", i
                        l = pair_indices(k, type1, i)
                        write(*,*) "l", l
                        n = n_all_pair_indices(k, type1, type2, i)
                        write(*,*) "n", n
                        do j = 1, n
                            m = all_pair_indices(k, type1, type2, i, j)
                            write(*,*) type1, type2, i, j
                            norm = distance_matrix(l, m)

                            pair_norm = atomic_charges(l) * atomic_charges(m) / norm &
                                & * cent_decay_matrix(k,l) * cent_decay_matrix(k, m) &
                                & * int_decay_matrix(l,m)

                            bag(nbag + j) = pair_norm
                            write(*,*) "end"
                        enddo
                        nbag = nbag + n
                    enddo
                    write(*,*) "end2"

                    ! sort
                    n = start_indices(type1, type2, 3)
                    do i = 1, nbag
                        write(*,*) i, nbag
                        l = maxloc(bag(:nbag), dim=1)
                        cm(k, n + i) = bag(l)
                        bag(l) = -huge_double
                    enddo
                    write(*,*) "end3"
                    ! end sort
                enddo
                write(*,*) "end4"
            enddo
            write(*,*) "end5"
        enddo
        write(*,*) "end6"

    ! Clean up
    deallocate(index_types)
    deallocate(n_pair_indices)
    deallocate(pair_indices)
    deallocate(n_all_pair_indices)
    deallocate(all_pair_indices)
    deallocate(distance_matrix)
    deallocate(cent_decay_matrix)
    deallocate(int_decay_matrix)
    deallocate(bag)
    deallocate(start_indices)
        write(*,*) "end7"

end subroutine fgenerate_local_bob

subroutine fgenerate_atomic_coulomb_matrix_sncf(atomic_charges, coordinates, natoms, nmax, &
        & cent_cutoff, cent_decay, int_cutoff, int_decay, localization, alt, cm)

    implicit none

    double precision, dimension(:), intent(in) :: atomic_charges
    double precision, dimension(:,:), intent(in) :: coordinates
    integer,intent(in) :: natoms
    integer, intent(in) :: nmax
    double precision, intent(inout) :: cent_cutoff, cent_decay, int_cutoff, int_decay
    integer, intent(in) :: localization
    logical, intent(in) :: alt

    double precision, dimension(natoms, ((nmax + 1) * nmax) / 2), intent(out):: cm

    integer :: idx

    double precision :: pair_norm
    double precision :: prefactor
    double precision :: norm
    double precision :: huge_double

    integer, allocatable, dimension(:, :) :: sorted_atoms_all
    integer, allocatable, dimension(:) :: cutoff_count

    double precision, allocatable, dimension(:, :, :) :: pair_distance_matrix
    double precision, allocatable, dimension(:, :) :: distance_matrix
    double precision, allocatable, dimension(:, :) :: distance_matrix_tmp

    integer i, j, m, n, k

    double precision, parameter :: pi = 4.0d0 * atan(1.0d0)

    if (size(coordinates, dim=1) /= size(atomic_charges, dim=1)) then
        write(*,*) "ERROR: Coulomb matrix generation"
        write(*,*) size(coordinates, dim=1), "coordinates, but", &
            & size(atomic_charges, dim=1), "atom_types!"
        stop
    endif

    ! Allocate temporary
    allocate(distance_matrix(natoms,natoms))
    allocate(cutoff_count(natoms))

    huge_double = huge(distance_matrix(1,1))

    if (cent_cutoff < 0) then
        cent_cutoff = huge_double
    endif

    if ((int_cutoff < 0) .OR. (int_cutoff > 2 * cent_cutoff)) then
        int_cutoff = 2 * cent_cutoff
    endif

    if (cent_decay < 0) then
        cent_decay = 0.0d0
    else if (cent_decay > cent_cutoff) then
        cent_decay = cent_cutoff
    endif

    if (int_decay < 0) then
        int_decay = 0.0d0
    else if (int_decay > int_cutoff) then
        int_decay = int_cutoff
    endif


    cutoff_count = 1

    !$OMP PARALLEL DO PRIVATE(norm) REDUCTION(+:cutoff_count)
    do i = 1, natoms
        distance_matrix(i, i) = 0.0d0
        do j = i+1, natoms
            norm = sqrt(sum((coordinates(j,:) - coordinates(i,:))**2))
            distance_matrix(i, j) = norm
            distance_matrix(j, i) = norm
            if (norm < cent_cutoff) then
                cutoff_count(i) = cutoff_count(i) + 1
                cutoff_count(j) = cutoff_count(j) + 1
            endif
        enddo
    enddo
    !$OMP END PARALLEL DO

    do i = 1, natoms
        if (cutoff_count(i) > nmax) then
            write(*,*) "ERROR: Coulomb matrix generation"
            write(*,*) nmax, "size set, but", &
                & cutoff_count(i), "size needed!"
            stop
        endif
    enddo

    ! Allocate temporary
    allocate(pair_distance_matrix(natoms, natoms, natoms))

    pair_distance_matrix = 0.0d0

    !$OMP PARALLEL DO PRIVATE(norm, pair_norm, prefactor)
    do i = 1, natoms
        pair_norm = 0.5d0 * atomic_charges(i) ** 2.4d0
        pair_distance_matrix(i,i,i) = pair_norm
        do k = 1, natoms
            ! self interaction
            if (distance_matrix(i,k) > cent_cutoff) then
                cycle
            endif

            prefactor = 1.0d0
            if (distance_matrix(i,k) > cent_cutoff - cent_decay) then
                prefactor = 0.5d0 * (cos(pi &
                    & * (distance_matrix(i,k) - cent_cutoff + cent_decay) / cent_decay) + 1)
            endif

            do j = i, natoms
                if (i == j .AND. i == k) then
                    cycle
                endif

                if (distance_matrix(j,k) > cent_cutoff) then
                    cycle
                endif

                if (distance_matrix(i,j) > int_cutoff) then
                    cycle
                endif

                norm = distance_matrix(i,k) + distance_matrix(j,k)
                if (alt) then
                    norm = norm + distance_matrix(i,j)
                endif

                pair_norm = prefactor * atomic_charges(i) * atomic_charges(j) &
                    & / norm ** localization

                if (distance_matrix(i,j) > int_cutoff - int_decay) then
                    pair_norm = pair_norm * 0.5d0 * (cos(pi &
                        & * (distance_matrix(i,j) - int_cutoff + int_decay) / int_decay) + 1)
                endif


                if (distance_matrix(j,k) > cent_cutoff - cent_decay) then
                    pair_norm = pair_norm * 0.5d0 * (cos(pi &
                        & * (distance_matrix(j,k) - cent_cutoff + cent_decay) / cent_decay) + 1)
                endif

                pair_distance_matrix(i, j, k) = pair_norm
                pair_distance_matrix(j, i, k) = pair_norm
            enddo
        enddo
    enddo
    !$OMP END PARALLEL DO

    ! Allocate temporary
    allocate(sorted_atoms_all(natoms, natoms))


    !$OMP PARALLEL DO PRIVATE(j)
        do k = 1, natoms
            !$OMP CRITICAL
                do i = 1, cutoff_count(k)
                    j = minloc(distance_matrix(:,k), dim=1)
                    sorted_atoms_all(i, k) = j
                    distance_matrix(j, k) = huge_double
                enddo
            !$OMP END CRITICAL
        enddo
    !$OMP END PARALLEL DO




    ! Fill coulomb matrix according to sorted row-2-norms
    cm = 0.0d0

    !$OMP PARALLEL DO PRIVATE(i, j, idx)
        do k = 1, natoms
            do m = 1, cutoff_count(k)
                i = sorted_atoms_all(m, k)
                idx = (m*m+m)/2 - m
                do n = 1, m
                    j = sorted_atoms_all(n, k)
                    cm(k, idx+n) = pair_distance_matrix(i,j,k)
                enddo
            enddo
        enddo
    !$OMP END PARALLEL DO


    ! Clean up
    deallocate(sorted_atoms_all)
    deallocate(pair_distance_matrix)

end subroutine fgenerate_atomic_coulomb_matrix_sncf

subroutine fgenerate_local_coulomb_matrix_sncf(atomic_charges, coordinates, natoms, nmax, &
        & cent_cutoff, cent_decay, int_cutoff, int_decay, localization, alt, cm)

    implicit none

    double precision, dimension(:), intent(in) :: atomic_charges
    double precision, dimension(:,:), intent(in) :: coordinates
    integer,intent(in) :: natoms
    integer, intent(in) :: nmax
    double precision, intent(inout) :: cent_cutoff, cent_decay, int_cutoff, int_decay
    integer, intent(in) :: localization
    logical, intent(in) :: alt

    double precision, dimension(natoms, ((nmax + 1) * nmax) / 2), intent(out):: cm

    integer :: idx

    double precision, allocatable, dimension(:, :) :: row_norms
    double precision :: pair_norm
    double precision :: prefactor
    double precision :: norm
    double precision :: huge_double

    integer, allocatable, dimension(:, :) :: sorted_atoms_all
    integer, allocatable, dimension(:) :: cutoff_count

    double precision, allocatable, dimension(:, :, :) :: pair_distance_matrix
    double precision, allocatable, dimension(:, :) :: distance_matrix
    double precision, allocatable, dimension(:, :) :: distance_matrix_tmp

    integer i, j, m, n, k

    double precision, parameter :: pi = 4.0d0 * atan(1.0d0)

    if (size(coordinates, dim=1) /= size(atomic_charges, dim=1)) then
        write(*,*) "ERROR: Coulomb matrix generation"
        write(*,*) size(coordinates, dim=1), "coordinates, but", &
            & size(atomic_charges, dim=1), "atom_types!"
        stop
    endif

    ! Allocate temporary
    allocate(distance_matrix(natoms,natoms))
    allocate(cutoff_count(natoms))

    huge_double = huge(distance_matrix(1,1))

    if (cent_cutoff < 0) then
        cent_cutoff = huge_double
    endif

    if ((int_cutoff < 0) .OR. (int_cutoff > 2 * cent_cutoff)) then
        int_cutoff = 2 * cent_cutoff
    endif

    if (cent_decay < 0) then
        cent_decay = 0.0d0
    else if (cent_decay > cent_cutoff) then
        cent_decay = cent_cutoff
    endif

    if (int_decay < 0) then
        int_decay = 0.0d0
    else if (int_decay > int_cutoff) then
        int_decay = int_cutoff
    endif


    cutoff_count = 1

    !$OMP PARALLEL DO PRIVATE(norm) REDUCTION(+:cutoff_count)
    do i = 1, natoms
        distance_matrix(i, i) = 0.0d0
        do j = i+1, natoms
            norm = sqrt(sum((coordinates(j,:) - coordinates(i,:))**2))
            distance_matrix(i, j) = norm
            distance_matrix(j, i) = norm
            if (norm < cent_cutoff) then
                cutoff_count(i) = cutoff_count(i) + 1
                cutoff_count(j) = cutoff_count(j) + 1
            endif
        enddo
    enddo
    !$OMP END PARALLEL DO

    do i = 1, natoms
        if (cutoff_count(i) > nmax) then
            write(*,*) "ERROR: Coulomb matrix generation"
            write(*,*) nmax, "size set, but", &
                & cutoff_count(i), "size needed!"
            stop
        endif
    enddo

    ! Allocate temporary
    allocate(pair_distance_matrix(natoms, natoms, natoms))
    allocate(row_norms(natoms, natoms))

    pair_distance_matrix = 0.0d0
    row_norms = 0.0d0

    !$OMP PARALLEL DO PRIVATE(norm, pair_norm, prefactor) REDUCTION(+:row_norms)
    do i = 1, natoms
        pair_norm = 0.5d0 * atomic_charges(i) ** 2.4d0
        pair_distance_matrix(i,i,i) = pair_norm
        row_norms(i,i) = row_norms(i,i) + pair_norm * pair_norm
        do k = 1, natoms
            ! self interaction
            if (distance_matrix(i,k) > cent_cutoff) then
                cycle
            endif

            prefactor = 1.0d0
            if (distance_matrix(i,k) > cent_cutoff - cent_decay) then
                prefactor = 0.5d0 * (cos(pi &
                    & * (distance_matrix(i,k) - cent_cutoff + cent_decay) / cent_decay) + 1)
            endif

            do j = i, natoms
                if (i == j .AND. i == k) then
                    cycle
                endif

                if (distance_matrix(j,k) > cent_cutoff) then
                    cycle
                endif

                if (distance_matrix(i,j) > int_cutoff) then
                    cycle
                endif

                norm = distance_matrix(i,k) + distance_matrix(j,k)
                if (alt) then
                    norm = norm + distance_matrix(i,j)
                endif

                pair_norm = prefactor * atomic_charges(i) * atomic_charges(j) &
                    & / norm ** localization

                if (distance_matrix(i,j) > int_cutoff - int_decay) then
                    pair_norm = pair_norm * 0.5d0 * (cos(pi &
                        & * (distance_matrix(i,j) - int_cutoff + int_decay) / int_decay) + 1)
                endif


                if (distance_matrix(j,k) > cent_cutoff - cent_decay) then
                    pair_norm = pair_norm * 0.5d0 * (cos(pi &
                        & * (distance_matrix(j,k) - cent_cutoff + cent_decay) / cent_decay) + 1)
                endif

                pair_distance_matrix(i, j, k) = pair_norm
                pair_distance_matrix(j, i, k) = pair_norm
                pair_norm = pair_norm * pair_norm
                row_norms(i,k) = row_norms(i,k) + pair_norm
                if (i == j) then
                    cycle
                endif
                row_norms(j,k) = row_norms(j,k) + pair_norm
            enddo
        enddo
    enddo
    !$OMP END PARALLEL DO

    ! Allocate temporary
    allocate(sorted_atoms_all(natoms, natoms))

    !$OMP PARALLEL DO
        do k = 1, natoms
            row_norms(k,k) = huge_double
        enddo
    !$OMP END PARALLEL DO

    !$OMP PARALLEL DO PRIVATE(j)
        do k = 1, natoms
            !$OMP CRITICAL
                do i = 1, cutoff_count(k)
                    j = maxloc(row_norms(:,k), dim=1)
                    sorted_atoms_all(i, k) = j
                    row_norms(j,k) = -huge_double
                enddo
            !$OMP END CRITICAL
        enddo
    !$OMP END PARALLEL DO

    ! Clean up
    deallocate(row_norms)



    ! Fill coulomb matrix according to sorted row-2-norms
    cm = 0.0d0

    !$OMP PARALLEL DO PRIVATE(i, j, idx)
        do k = 1, natoms
            do m = 1, cutoff_count(k)
                i = sorted_atoms_all(m, k)
                idx = (m*m+m)/2 - m
                do n = 1, m
                    j = sorted_atoms_all(n, k)
                    cm(k, idx+n) = pair_distance_matrix(i,j,k)
                enddo
            enddo
        enddo
    !$OMP END PARALLEL DO


    ! Clean up
    deallocate(sorted_atoms_all)
    deallocate(pair_distance_matrix)

end subroutine fgenerate_local_coulomb_matrix_sncf
