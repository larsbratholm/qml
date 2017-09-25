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

    do j = 1, natoms
        if (nuclear_charges(j) == type1) then
            n = n + 1
            type1_indices(n) = j
        endif
    enddo

end subroutine get_indices

subroutine global_checks(nuclear_charges, coordinates, charge_types, nsize, &
        & potential_types, sorting, nasize, natypes, nrep, start_indices)

    implicit none

    integer, dimension(:), intent(in) :: nuclear_charges
    double precision, dimension(:,:), intent(in) :: coordinates
    integer, dimension(:), intent(in) :: charge_types
    integer, intent(in) :: nsize
    integer, dimension(:), intent(in) :: potential_types
    integer, intent(in) :: sorting
    integer, dimension(:), intent(in) :: nasize
    integer, intent(in) :: natypes
    integer, intent(in) :: nrep
    integer, dimension(:,:), intent(out) :: start_indices

    integer :: i, j, k, l, m, n

    ! size checks
    if (size(coordinates, dim=1) /= size(nuclear_charges, dim=1)) then
        write(*,*) "ERROR: Generation of global representation"
        write(*,*) size(coordinates, dim=1), "coordinates, but", &
            & size(nuclear_charges, dim=1), "atom_types!"
        stop
    endif

    if (size(charge_types, dim=1) /= size(nasize, dim=1)) then
        write(*,*) "ERROR: Generation of global representation"
        write(*,*) size(charge_types, dim=1), "unique atom types, but", &
            & size(nasize, dim=1), "max size!"
        stop
    endif

    if (potential_types(1) < 0 .OR. potential_types(1) > 1 .OR. &
        & potential_types(2) < 0 .OR. potential_types(2) > 1 .OR. & 
        & potential_types(3) < 0 .OR. potential_types(3) > 9 .OR. &
        & potential_types(4) < 0 .OR. potential_types(4) > 8) then
        write(*,*) "ERROR: Generation of global representation"
        write(*,*) "Unknown potential given"
        stop
    endif

    ! check size of representation
    if (sorting > 0 .AND. sorting < 4) then
        n = 0
        start_indices(1,1) = n
        if (potential_types(1) == 1) then
            n = n + nsize
        endif
        start_indices(2,1) = n
        if (potential_types(2) == 1) then
            n = n + (nsize * (nsize - 1)) / 2
        endif
        start_indices(3,1) = n
        if (potential_types(3) > 0 .AND. potential_types(3) < 5) then
            n = n + (nsize * (nsize - 1) * (nsize - 2)) / 2
        endif
        if (potential_types(3) > 4) then
            n = n + (nsize * (nsize - 1) * (nsize - 2)) / 6
        endif
        start_indices(4,1) = n
        if (potential_types(4) > 0 .AND. potential_types(4) < 3) then
            n = n + (nsize * (nsize - 1) * (nsize - 2) * (nsize - 3)) / 2
        endif
        if (potential_types(4) > 2 .AND. potential_types(4) < 7) then
            n = n + (nsize * (nsize - 1) * (nsize - 2) * (nsize - 3)) / 6
        endif
        if (potential_types(4) > 6) then
            n = n + (nsize * (nsize - 1) * (nsize - 2) * (nsize - 3)) / 24
        endif
    else if (sorting == 4) then
        n = 0
        if (potential_types(1) == 1) then
            ! A bag
            do i=1, natypes
                start_indices(1,i) = n
                n = n + nasize(i)
            enddo
        endif
        if (potential_types(2) == 1) then
            do i=1, natypes
                ! AA bag
                start_indices(2,(i*(i+1))/2) = n
                n = n + (nasize(i) * (nasize(i) - 1)) / 2
                ! AB bag
                do j=1, i-1
                    start_indices(2,(i*(i-1))/2 + j) = n
                    n = n + nasize(i) * nasize(j)
                enddo
            enddo
        endif
        if (potential_types(3) > 0 .AND. potential_types(3) < 5) then
            m = 0
            do i=1, natypes
                ! AAA bag
                m = m+1
                start_indices(3,m) = n
                n = n + (nasize(i) * (nasize(i) - 1) * (nasize(i) - 2)) / 2
                do j=1, natypes
                    if (j == i) cycle
                    ! AAB bag
                    m = m+1
                    start_indices(3,m) = n
                    n = n + nasize(i) * (nasize(i) - 1) * nasize(j)
                    ! ABB bag
                    m = m+1
                    start_indices(3,m) = n
                    n = n + nasize(i) * (nasize(j) * (nasize(j)-1)) / 2
                    ! ABC bag
                    do k=j+1, natypes
                        if (k == i) cycle
                        m = m+1
                        start_indices(3,m) = n
                        n = n + nasize(i) * nasize(j) * nasize(k)
                    enddo
                enddo
            enddo
        else if (potential_types(3) > 4) then
            m = 0
            do i=1, natypes
                ! AAA bag
                m = m+1
                start_indices(3,m) = n
                n = n + (nasize(i) * (nasize(i) - 1) * (nasize(i) - 2)) / 6
                do j=i+1, natypes
                    ! AAB bag
                    m = m+1
                    start_indices(3,m) = n
                    n = n + (nasize(i) * (nasize(i) - 1))/2 * nasize(j)
                    ! ABB bag
                    m = m+1
                    start_indices(3,m) = n
                    n = n + nasize(i) * (nasize(j) * (nasize(j)-1)) / 2
                    ! ABC bag
                    do k=j+1, natypes
                        m = m+1
                        start_indices(3,m) = n
                        n = n + nasize(i) * nasize(j) * nasize(k)
                    enddo
                enddo
            enddo
        endif
        if (potential_types(4) > 0 .AND. potential_types(4) < 3) then
            m = 0
            do i=1, natypes
                ! AAAA bag
                m = m+1
                start_indices(4,m) = n
                n = n + (nasize(i) * (nasize(i) - 1) * (nasize(i) - 2) &
                    & * (nasize(i) - 3)) / 2
                do j=1, natypes
                    if (j == i) cycle
                    ! AAAB bag
                    m = m+1
                    start_indices(4,m) = n
                    n = n + nasize(i) * (nasize(i) - 1) * (nasize(i) - 2) * nasize(j)
                    ! AABB bag
                    m = m+1
                    start_indices(4,m) = n
                    n = n + nasize(i) * (nasize(i) - 1) * (nasize(j) * (nasize(j)-1)) / 2
                    ! ABBB bag
                    m = m+1
                    start_indices(4,m) = n
                    n = n + nasize(i) * (nasize(j) * (nasize(j) - 1) * (nasize(j) - 2)) / 2
                    ! ABAA, 
                    m = m+1
                    start_indices(4,m) = n
                    n = n + nasize(i) * nasize(j) * ((nasize(i)-1) * (nasize(i) - 2)) / 2
                    ! ABAB, 
                    m = m+1
                    start_indices(4,m) = n
                    n = n + nasize(i) * nasize(j) * (nasize(i) -1) * (nasize(j) -1)
                    do k = 1, natypes
                        if (k == i .OR. k == j) cycle
                        ! AABC, 
                        m = m+1
                        start_indices(4,m) = n
                        n = nasize(i) * (nasize(i)-1) * nasize(j) * nasize(k)
                        ! ABAC, 
                        m = m+1
                        start_indices(4,m) = n
                        n = nasize(i) * nasize(j) * (nasize(i)-1) * nasize(k)
                        ! ABBC, 
                        m = m+1
                        start_indices(4,m) = n
                        n = nasize(i) * nasize(j) * (nasize(j)-1) * nasize(k)
                        ! ABCC
                        m = m+1
                        start_indices(4,m) = n
                        n = nasize(i) * nasize(j) * (nasize(k) * (nasize(k)-1))/2
                        ! ABCD
                        do l = 1, natypes
                            if (l == i .OR. l == j .OR. l == k) cycle
                            m = m+1
                            start_indices(4,m) = n
                            n = nasize(i) * nasize(j) * nasize(k) * nasize(l)
                        enddo
                    enddo
                enddo
            enddo
        else if (potential_types(4) > 2 .AND. potential_types(4) < 7) then
            m = 0
            do i=1, natypes
                ! AAAA bag
                m = m+1
                start_indices(4,m) = n
                n = n + (nasize(i) * (nasize(i) - 1) * (nasize(i) - 2) &
                    & * (nasize(i) - 3)) / 6
                do j=1, natypes
                    if (j == i) cycle
                    ! AAAB bag
                    m = m+1
                    start_indices(4,m) = n
                    n = n + nasize(i) * ((nasize(i) - 1) * (nasize(i) - 2))/2 * nasize(j)
                    ! AABB bag
                    m = m+1
                    start_indices(4,m) = n
                    n = n + nasize(i) * (nasize(i) - 1) * (nasize(j) * (nasize(j)-1)) / 2
                    ! ABBB bag
                    m = m+1
                    start_indices(4,m) = n
                    n = n + nasize(i) * (nasize(j) * (nasize(j) - 1) * (nasize(j) - 2)) / 6
                    do k = j+1, natypes
                        if (k == i) cycle
                        ! AABC, 
                        m = m+1
                        start_indices(4,m) = n
                        n = nasize(i) * (nasize(i)-1) * nasize(j) * nasize(k)
                        ! ABBC, 
                        m = m+1
                        start_indices(4,m) = n
                        n = nasize(i) * (nasize(j) * (nasize(j)-1))/2 * nasize(k)
                        ! ABCD
                        do l = k+1, natypes
                            if (l == i) cycle
                            m = m+1
                            start_indices(4,m) = n
                            n = nasize(i) * nasize(j) * nasize(k) * nasize(l)
                        enddo
                    enddo
                enddo
            enddo
        else if (potential_types(4) > 6) then
            m = 0
            do i=1, natypes
                ! AAAA bag
                m = m+1
                start_indices(4,m) = n
                n = n + (nasize(i) * (nasize(i) - 1) * (nasize(i) - 2) &
                    & * (nasize(i) - 3)) / 24
                do j=i+1, natypes
                    ! AAAB bag
                    m = m+1
                    start_indices(4,m) = n
                    n = n + nasize(i) * ((nasize(i) - 1) * (nasize(i) - 2))/6 * nasize(j)
                    ! AABB bag
                    m = m+1
                    start_indices(4,m) = n
                    n = n + (nasize(i) * (nasize(i) - 1))/2 * (nasize(j) * (nasize(j)-1)) / 2
                    do k = j+1, natypes
                        ! AABC, 
                        m = m+1
                        start_indices(4,m) = n
                        n = (nasize(i) * (nasize(i)-1))/2 * nasize(j) * nasize(k)
                        ! ABCD
                        do l = k+1, natypes
                            m = m+1
                            start_indices(4,m) = n
                            n = nasize(i) * nasize(j) * nasize(k) * nasize(l)
                        enddo
                    enddo
                enddo
            enddo
        endif
    else
        write(*,*) "ERROR: Generation of global representation"
        write(*,*) "Unknown sorting scheme:", sorting, "given"
    endif

    if (n /= nrep) then
        write(*,*) "ERROR: Generation of global representation"
        write(*,*) "Inferred vector size", n, "but given size", nrep
        stop
    endif

end subroutine global_checks

subroutine get_index_types(charge_types, natypes, natoms, nuclear_charges, index_types)

    implicit none

    integer, dimension(:), intent(in) :: charge_types
    integer, intent(in) :: natypes
    integer, intent(in) :: natoms
    integer, dimension(:), intent(in) :: nuclear_charges
    integer, dimension(:), intent(out) :: index_types

    integer :: i,j,type1

    index_types = 1

    ! Get element types
    if (natypes > 2) then
        do i = 1, natoms
            type1 = nuclear_charges(i)
            do j = 2, natypes
                if (type1 == charge_types(j)) then
                    index_types(i) = j
                    exit
                endif
            enddo
        enddo
    endif

end subroutine get_index_types

subroutine get_distance_matrix(coordinates, distance_matrix)

    implicit none

    double precision, dimension(:,:), intent(in) :: coordinates
    double precision, dimension(:,:), intent(out) :: distance_matrix

    integer :: i,j
    double precision :: norm

    do i = 1, size(coordinates, dim=1)
        distance_matrix(i, i) = 0.0d0
        do j = 1, i-1
            norm = sqrt(sum((coordinates(i,:) - coordinates(j,:))**2))
            distance_matrix(i, j) = norm
            distance_matrix(j, i) = norm
        enddo
    enddo
end subroutine get_distance_matrix

subroutine smoothen_distance_matrix(natypes, natoms, charge_types, index_types, distance_matrix)

    implicit none

    integer, intent(in) :: natypes
    integer, intent(in) :: natoms
    integer, dimension(:), intent(in) :: charge_types
    integer, dimension(:), intent(in) :: index_types
    double precision, dimension(:,:), intent(inout) :: distance_matrix

    double precision, allocatable, dimension(:,:) :: V
    double precision, allocatable, dimension(:,:) :: V_matrix
    integer :: i, j, k, l

    ! allocate temporary
    allocate(V(natypes,natypes))

    do i=1, natypes
        V(i,i) = 2/charge_types(i)**0.4d0
    enddo

    do i=1, natypes
        do j=i+1, natypes
            V(i,j) = sqrt((V(i,i)**2 + V(j,j)**2)/2)
            V(j,i) = V(i,j)
        enddo
    enddo

    ! allocate temporary
    allocate(V_matrix(natoms,natoms))

    do i=1, natoms
        k = index_types(i)
        V_matrix(i,i) = V(k,k)
        do j=i+1, natoms
            l = index_types(j)
            V_matrix(i,j) = V(k,l)
            V_matrix(j,i) = V(k,l)
        enddo
    enddo

    distance_matrix = ((distance_matrix**3 + V_matrix**3))**(1.0d0/3)

    ! deallocate
    deallocate(V)
    deallocate(V_matrix)

end subroutine smoothen_distance_matrix

subroutine get_convenience_arrays_global(distance_matrix, index_types, cutoff, decay_matrix, natypes, &
        & natoms, charge_types, nasize, ntype_indices, type_indices, decay, mask)

    implicit none

    double precision, dimension(:,:), intent(in) :: distance_matrix
    integer, dimension(:), intent(in) :: index_types
    double precision, intent(in) :: cutoff
    double precision, intent(in) :: decay
    integer, intent(in) :: natoms
    integer, intent(in) :: natypes
    integer, dimension(:), intent(in) :: charge_types
    integer, dimension(:), intent(in) :: nasize

    integer, dimension(:), intent(out) :: ntype_indices
    integer, dimension(:,:), intent(out) :: type_indices
    logical, dimension(:,:), intent(out) :: mask
    double precision, dimension(:,:), intent(out) :: decay_matrix

    integer :: i, j, type1, n
    double precision :: norm, decay_dist
    integer, allocatable, dimension(:) :: type_count

    double precision, parameter :: pi = 4.0d0 * atan(1.0d0)

    !n_pair_indices = 0
    mask = .false.
    decay_matrix = 1.0d0
    ntype_indices = 0

    ! Create convenience arrays
    do i = 1, natoms
        do j = 1, i-1
            norm = distance_matrix(i,j)
            if (norm < cutoff) then
                ! Calculate decay factors
                decay_dist = norm - cutoff + decay
                if (decay_dist > 0) then
                    decay_matrix(i,j) = 0.5d0 * (cos(pi * decay_dist / decay) + 1)
                    decay_matrix(j,i) = decay_matrix(i,j)
                endif
            else
                mask(i,j) = .true.
                mask(j,i) = .true.
            endif
        enddo
    enddo

    do i = 1, natoms
        type1 = index_types(i)
        n = ntype_indices(type1) + 1
        ntype_indices(type1) = n
        type_indices(type1, n) = i
    enddo

    ! Check given array sizes
    do j=1, natypes
        if (ntype_indices(j) > nasize(j)) then
            write(*,*) "ERROR: Bag of Bonds generation"
            write(*,*) nasize(j), "size set for atom with nuclear charge,", &
                & charge_types(j), "but", ntype_indices(j), "size needed!"
        endif
    enddo

end subroutine get_convenience_arrays_global

subroutine cm_global_three_body(three_body, potential_type, natoms, nuclear_charges, &
        & mask, decay_matrix, distance_matrix, localization, coordinates)

    implicit none

    double precision, dimension(:,:,:), intent(inout) :: three_body
    integer, intent(in) :: potential_type
    integer, intent(in) :: natoms
    integer, dimension(:), intent(in) :: nuclear_charges
    logical, dimension(:,:), intent(in) :: mask
    double precision, dimension(:,:), intent(in) :: decay_matrix
    double precision, dimension(:,:), intent(in) :: distance_matrix
    integer, intent(in) :: localization
    double precision, dimension(:,:), intent(in) :: coordinates

    integer :: i, j, k
    double precision :: norm, one_third, rnorm
    double precision, allocatable, dimension(:) :: r

    one_third = (1.0d0/3)

    if (potential_type == 1) then
        ! 1: centered on 1: Z2*Z3 / (R12 + R13)
        !$OMP PARALLEL DO PRIVATE(norm) COLLAPSE(2) SCHEDULE(dynamic)
        do i=1, natoms
            do j=1, natoms
                if (i == j) cycle
                if (mask(i,j)) cycle
                do k=j+1, natoms
                    if (i == k) cycle
                    if (mask(i,k) .OR. mask(j,k)) cycle

                    norm = one_third * nuclear_charges(j) * nuclear_charges(k) / &
                        & (distance_matrix(i,j) + distance_matrix(i,k)) ** localization * &
                        & decay_matrix(i,k) * decay_matrix(i,j) * decay_matrix(j,k)

                    three_body(i,j,k) = norm
                    three_body(i,k,j) = norm
                enddo
            enddo
        enddo
        !$OMP END PARALLEL DO
    else if (potential_type == 2) then
        ! 2: centered on 1: Z2*Z3 / (R12 + R13 + R23)
        !$OMP PARALLEL DO PRIVATE(norm) COLLAPSE(2) SCHEDULE(dynamic)
        do i=1, natoms
            do j=1, natoms
                if (i == j) cycle
                if (mask(i,j)) cycle
                do k=j+1, natoms
                    if (i == k) cycle
                    if (mask(i,k) .OR. mask(j,k)) cycle

                    norm = one_third * nuclear_charges(j) * nuclear_charges(k) / &
                        & (distance_matrix(i,j) + distance_matrix(i,k) + distance_matrix(j,k)) ** &
                        & localization * decay_matrix(i,k) * decay_matrix(i,j) * decay_matrix(j,k)

                    three_body(i,j,k) = norm
                    three_body(i,k,j) = norm
                enddo
            enddo
        enddo
        !$OMP END PARALLEL DO
    else if (potential_type == 3) then
        ! 3: centered on 1: Z1*(Z2+Z3)/|r13 - m2/(m2+m3) * r23|

        ! Allocate temporary
        allocate(r(3))

        !$OMP PARALLEL DO PRIVATE(rnorm, norm, r) COLLAPSE(2) SCHEDULE(dynamic)
        do i=1, natoms
            do j=1, natoms
                if (i == j) cycle
                if (mask(i,j)) cycle
                do k=j+1, natoms
                    if (i == k) cycle
                    if (mask(i,k) .OR. mask(j,k)) cycle

                    r = (coordinates(i,:) - coordinates(k,:)) - &
                        & nuclear_charges(j) / (nuclear_charges(j) + nuclear_charges(k)) * &
                        & (coordinates(j,:) - coordinates(k,:))
                    rnorm = sqrt(sum(r**2))
                    norm = one_third * nuclear_charges(i) * (nuclear_charges(j) + nuclear_charges(k)) / &
                        & rnorm * decay_matrix(i,k) * decay_matrix(i,j) * decay_matrix(j,k)

                    three_body(i,j,k) = norm
                    three_body(i,k,j) = norm
                enddo
            enddo
        enddo
        !$OMP END PARALLEL DO

        ! deallocate
        deallocate(r)
    else if (potential_type == 4) then
        ! 4: centered on 1: Z2*Z3/R23 + Z1*(Z2+Z3)/(r13 - m2/(m2+m3) * r23)

        ! Allocate temporary
        allocate(r(3))

        !$OMP PARALLEL DO PRIVATE(rnorm, norm, r) COLLAPSE(2) SCHEDULE(dynamic)
        do i=1, natoms
            do j=1, natoms
                if (i == j) cycle
                if (mask(i,j)) cycle
                do k=j+1, natoms
                    if (i == k) cycle
                    if (mask(i,k) .OR. mask(j,k)) cycle

                    r = (coordinates(i,:) - coordinates(k,:)) - &
                        & nuclear_charges(j) / (nuclear_charges(j) + nuclear_charges(k)) * &
                        & (coordinates(j,:) - coordinates(k,:))
                    rnorm = sqrt(sum(r**2))
                    norm = one_third * (nuclear_charges(i) * (nuclear_charges(j) + nuclear_charges(k)) / &
                        & rnorm + nuclear_charges(j) * nuclear_charges(k) / distance_matrix(j,k)) * &
                        & decay_matrix(i,k) * decay_matrix(i,j) * decay_matrix(j,k)

                    three_body(i,j,k) = norm
                    three_body(i,k,j) = norm
                enddo
            enddo
        enddo
        !$OMP END PARALLEL DO

        ! deallocate
        deallocate(r)
    else if (potential_type == 5) then
        ! 5: (1) average
        ! 1: centered on 1: Z2*Z3 / (R12 + R13)

        !$OMP PARALLEL DO PRIVATE(norm) SCHEDULE(dynamic)
        do i=1, natoms
            do j=i+1, natoms
                if (mask(i,j)) cycle
                do k=j+1, natoms
                    if (mask(i,k) .OR. mask(j,k)) cycle

                    norm = nuclear_charges(j) * nuclear_charges(k) / &
                        & (distance_matrix(i,j) + distance_matrix(i,k)) ** &
                        & localization

                    norm = norm + nuclear_charges(i) * nuclear_charges(k) / &
                        & (distance_matrix(i,j) + distance_matrix(j,k)) ** &
                        & localization

                    norm = norm + nuclear_charges(j) * nuclear_charges(i) / &
                        & (distance_matrix(j,k) + distance_matrix(i,k)) ** &
                        & localization

                    norm = one_third * norm * decay_matrix(i,k) * decay_matrix(j,k) * decay_matrix(i,j)

                    three_body(i,j,k) = norm
                    three_body(i,k,j) = norm
                    three_body(j,i,k) = norm
                    three_body(j,k,i) = norm
                    three_body(k,i,j) = norm
                    three_body(k,j,i) = norm
                enddo
            enddo
        enddo
        !$OMP END PARALLEL DO
    else if (potential_type == 6) then
        ! 6: (2) average
        ! 2: centered on 1: Z2*Z3 / (R12 + R13 + R23)

        !$OMP PARALLEL DO PRIVATE(norm) SCHEDULE(dynamic)
        do i=1, natoms
            do j=i+1, natoms
                if (mask(i,j)) cycle
                do k=j+1, natoms
                    if (mask(i,k) .OR. mask(j,k)) cycle

                    norm = one_third * (nuclear_charges(j) * nuclear_charges(k) + &
                        & nuclear_charges(i) * nuclear_charges(k) + &
                        & nuclear_charges(i) * nuclear_charges(j)) / &
                        & (distance_matrix(i,j) + distance_matrix(i,k) + distance_matrix(j,k)) ** &
                        & localization * decay_matrix(i,k) * decay_matrix(i,j) * decay_matrix(j,k)


                    three_body(i,j,k) = norm
                    three_body(i,k,j) = norm
                    three_body(j,i,k) = norm
                    three_body(j,k,i) = norm
                    three_body(k,i,j) = norm
                    three_body(k,j,i) = norm
                enddo
            enddo
        enddo
        !$OMP END PARALLEL DO
    else if (potential_type == 7) then
        ! 7: (3) average
        ! 3: centered on 1: Z1*(Z2+Z3)/|r13 - m2/(m2+m3) * r23|

        ! Allocate temporary
        allocate(r(3))

        !$OMP PARALLEL DO PRIVATE(rnorm, norm, r) SCHEDULE(dynamic)
        do i=1, natoms
            do j=i+1, natoms
                if (mask(i,j)) cycle
                do k=j+1, natoms
                    if (mask(i,k) .OR. mask(j,k)) cycle

                    r = (coordinates(i,:) - coordinates(k,:)) - &
                        & nuclear_charges(j) / (nuclear_charges(j) + nuclear_charges(k)) * &
                        & (coordinates(j,:) - coordinates(k,:))
                    rnorm = sqrt(sum(r**2))

                    norm = nuclear_charges(i) * (nuclear_charges(j) + nuclear_charges(k)) / &
                        & rnorm 

                    r = (coordinates(j,:) - coordinates(k,:)) - &
                        & nuclear_charges(i) / (nuclear_charges(i) + nuclear_charges(k)) * &
                        & (coordinates(i,:) - coordinates(k,:))
                    rnorm = sqrt(sum(r**2))

                    norm = norm + nuclear_charges(j) * (nuclear_charges(i) + nuclear_charges(k)) / &
                        & rnorm 

                    r = (coordinates(k,:) - coordinates(j,:)) - &
                        & nuclear_charges(i) / (nuclear_charges(i) + nuclear_charges(j)) * &
                        & (coordinates(i,:) - coordinates(j,:))
                    rnorm = sqrt(sum(r**2))

                    norm = norm + nuclear_charges(k) * (nuclear_charges(i) + nuclear_charges(j)) / &
                        & rnorm 

                    norm = norm * one_third * decay_matrix(i,k) * decay_matrix(i,j) * decay_matrix(j,k)

                    three_body(i,j,k) = norm
                    three_body(i,k,j) = norm
                    three_body(j,i,k) = norm
                    three_body(j,k,i) = norm
                    three_body(k,i,j) = norm
                    three_body(k,j,i) = norm
                enddo
            enddo
        enddo
        !$OMP END PARALLEL DO

        ! deallocate
        deallocate(r)
    else if (potential_type == 8) then
        ! 8: (4) average
        ! 4: centered on 1: Z2*Z3/R23 + Z1*(Z2+Z3)/(r13 - m2/(m2+m3) * r23)

        ! Allocate temporary
        allocate(r(3))

        !$OMP PARALLEL DO PRIVATE(rnorm, norm, r) SCHEDULE(dynamic)
        do i=1, natoms
            do j=i+1, natoms
                if (mask(i,j)) cycle
                do k=j+1, natoms
                    if (mask(i,k) .OR. mask(j,k)) cycle

                    r = (coordinates(i,:) - coordinates(k,:)) - &
                        & nuclear_charges(j) / (nuclear_charges(j) + nuclear_charges(k)) * &
                        & (coordinates(j,:) - coordinates(k,:))
                    rnorm = sqrt(sum(r**2))

                    norm = nuclear_charges(i) * (nuclear_charges(j) + nuclear_charges(k)) / &
                        & rnorm 

                    r = (coordinates(j,:) - coordinates(k,:)) - &
                        & nuclear_charges(i) / (nuclear_charges(i) + nuclear_charges(k)) * &
                        & (coordinates(i,:) - coordinates(k,:))
                    rnorm = sqrt(sum(r**2))

                    norm = norm + nuclear_charges(j) * (nuclear_charges(i) + nuclear_charges(k)) / &
                        & rnorm 

                    r = (coordinates(k,:) - coordinates(j,:)) - &
                        & nuclear_charges(i) / (nuclear_charges(i) + nuclear_charges(j)) * &
                        & (coordinates(i,:) - coordinates(j,:))
                    rnorm = sqrt(sum(r**2))

                    norm = norm + nuclear_charges(k) * (nuclear_charges(i) + nuclear_charges(j)) / &
                        & rnorm 

                    norm = norm + nuclear_charges(i) * nuclear_charges(j) / distance_matrix(i,j) + &
                        & nuclear_charges(i) * nuclear_charges(k) / distance_matrix(i,k) + &
                        & nuclear_charges(j) * nuclear_charges(k) / distance_matrix(j,k)

                    norm = norm * one_third * decay_matrix(i,k) * decay_matrix(i,j) * decay_matrix(j,k)

                    three_body(i,j,k) = norm
                    three_body(i,k,j) = norm
                    three_body(j,i,k) = norm
                    three_body(j,k,i) = norm
                    three_body(k,i,j) = norm
                    three_body(k,j,i) = norm

                enddo
            enddo
        enddo
        !$OMP END PARALLEL DO

        ! deallocate
        deallocate(r)
    else if (potential_type == 9) then
        ! 9: (Z1*Z2*Z3/(Z1+Z2+Z3)) * (3 * cos(THETA123) * cos(THETA312) * cos(THETA231) + 1) / (R12*R13*R23)**3

        !$OMP PARALLEL DO PRIVATE(rnorm, norm, r) SCHEDULE(dynamic)
        do i=1, natoms
            do j=i+1, natoms
                if (mask(i,j)) cycle
                do k=j+1, natoms
                    if (mask(i,k) .OR. mask(j,k)) cycle

                    norm = distance_matrix(i,j) * distance_matrix(i,k) * distance_matrix(j,k)
                    norm = (nuclear_charges(i) * nuclear_charges(j) * nuclear_charges(k)) / &
                        & (nuclear_charges(i) + nuclear_charges(j) + nuclear_charges(k)) * &
                        & (1/norm + &
                        & dot_product(coordinates(j,:) - coordinates(k,:), &
                        & coordinates(i,:) - coordinates(k,:)) * &
                        & dot_product(coordinates(i,:) - coordinates(j,:), &
                        & coordinates(k,:) - coordinates(j,:)) * &
                        & dot_product(coordinates(k,:) - coordinates(i,:), &
                        & coordinates(j,:) - coordinates(i,:)) / &
                    & norm * decay_matrix(i,k) * decay_matrix(i,j) * decay_matrix(j,k))

                    three_body(i,j,k) = norm
                    three_body(i,k,j) = norm
                    three_body(j,i,k) = norm
                    three_body(j,k,i) = norm
                    three_body(k,i,j) = norm
                    three_body(k,j,i) = norm
                enddo
            enddo
        enddo
        !$OMP END PARALLEL DO

        ! deallocate
        deallocate(r)
    endif

end subroutine cm_global_three_body

subroutine cm_global_four_body(four_body, potential_type, natoms, nuclear_charges, &
        & mask, decay_matrix, distance_matrix)

    implicit none

    double precision, dimension(:,:,:,:), intent(inout) :: four_body
    integer, intent(in) :: potential_type
    integer, intent(in) :: natoms
    integer, dimension(:), intent(in) :: nuclear_charges
    logical, dimension(:,:), intent(in) :: mask
    double precision, dimension(:,:), intent(in) :: decay_matrix
    double precision, dimension(:,:), intent(in) :: distance_matrix

    integer :: i, j, k,l
    double precision :: norm, one_twelfth, rnorm
    double precision, allocatable, dimension(:) :: r

    one_twelfth = (1.0d0/12)

    if (potential_type == 1) then
        ! 1: centered on 1,2: = 
        !    Z1*Z2/R12 + R12*Z2*Z3/((R12+R13)*R23) + R12*Z2*Z4/((R12+R14)*R24)
        !$OMP PARALLEL DO PRIVATE(norm) COLLAPSE(2) SCHEDULE(dynamic)
        do i=1, natoms
            do j=1, natoms
                if (i == j) cycle
                if (mask(i,j)) cycle
                do k=1, natoms
                    if (i == k .OR. j == k) cycle
                    if (mask(i,k) .OR. mask(j,k)) cycle
                    do l=k+1, natoms
                        if (l == i .OR. l == j) cycle
                        if (mask(i,l) .OR. mask(j,l) .OR. mask(k,l)) cycle

                        norm = nuclear_charges(i) * nuclear_charges(j) / distance_matrix(i,j) + &
                            & distance_matrix(i,j) * nuclear_charges(j) * nuclear_charges(k) / &
                            & ((distance_matrix(i,j)+distance_matrix(i,k))*distance_matrix(j,k)) + &
                            & distance_matrix(i,j) * nuclear_charges(j) * nuclear_charges(l) / &
                            & ((distance_matrix(i,j)+distance_matrix(i,l))*distance_matrix(j,l))

                        norm = one_twelfth * norm * decay_matrix(i,j) * decay_matrix(i,k) * decay_matrix(i,l) * &
                            & decay_matrix(j,k) * decay_matrix(j,l) * decay_matrix(k,l)

                        four_body(i,j,k,l) = norm
                        four_body(i,j,l,k) = norm
                    enddo
                enddo
            enddo
        enddo
        !$OMP END PARALLEL DO
    else if (potential_type == 2) then
        ! 2: centered on 1,2: = R12*Z2*Z3/((R12+R13)*R23) + R12*Z2*Z4/((R12+R14)*R24)
        !$OMP PARALLEL DO PRIVATE(norm) COLLAPSE(2) SCHEDULE(dynamic)
        do i=1, natoms
            do j=1, natoms
                if (i == j) cycle
                if (mask(i,j)) cycle
                do k=1, natoms
                    if (i == k .OR. j == k) cycle
                    if (mask(i,k) .OR. mask(j,k)) cycle
                    do l=k+1, natoms
                        if (l == i .OR. l == j) cycle
                        if (mask(i,l) .OR. mask(j,l) .OR. mask(k,l)) cycle

                        norm = distance_matrix(i,j) * nuclear_charges(j) * nuclear_charges(k) / &
                            & ((distance_matrix(i,j)+distance_matrix(i,k))*distance_matrix(j,k)) + &
                            & distance_matrix(i,j) * nuclear_charges(j) * nuclear_charges(l) / &
                            & ((distance_matrix(i,j)+distance_matrix(i,l))*distance_matrix(j,l))

                        norm = one_twelfth * norm * decay_matrix(i,j) * decay_matrix(i,k) * decay_matrix(i,l) * &
                            & decay_matrix(j,k) * decay_matrix(j,l) * decay_matrix(k,l)

                        four_body(i,j,k,l) = norm
                        four_body(i,j,l,k) = norm
                    enddo
                enddo
            enddo
        enddo
        !$OMP END PARALLEL DO
    else if (potential_type == 3) then
        ! 1: centered on 1, averaged over 2
        !    Z1*Z2/R12 + R12*Z2*Z3/((R12+R13)*R23) + R12*Z2*Z4/((R12+R14)*R24)
        !$OMP PARALLEL DO PRIVATE(norm) COLLAPSE(2) SCHEDULE(dynamic)
        do i=1, natoms
            do j=1, natoms
                if (i == j) cycle
                if (mask(i,j)) cycle
                do k=j+1, natoms
                    if (i == k) cycle
                    if (mask(i,k) .OR. mask(j,k)) cycle
                    do l=k+1, natoms
                        if (l == i) cycle
                        if (mask(i,l) .OR. mask(j,l) .OR. mask(k,l)) cycle

                        norm = nuclear_charges(i) * nuclear_charges(j) / distance_matrix(i,j) + &
                            & distance_matrix(i,j) * nuclear_charges(j) * nuclear_charges(k) / &
                            & ((distance_matrix(i,j)+distance_matrix(i,k))*distance_matrix(j,k)) + &
                            & distance_matrix(i,j) * nuclear_charges(j) * nuclear_charges(l) / &
                            & ((distance_matrix(i,j)+distance_matrix(i,l))*distance_matrix(j,l))

                        norm = norm + nuclear_charges(i) * nuclear_charges(k) / distance_matrix(i,k) + &
                            & distance_matrix(i,k) * nuclear_charges(k) * nuclear_charges(j) / &
                            & ((distance_matrix(i,k)+distance_matrix(i,j))*distance_matrix(j,k)) + &
                            & distance_matrix(i,k) * nuclear_charges(k) * nuclear_charges(l) / &
                            & ((distance_matrix(i,k)+distance_matrix(i,l))*distance_matrix(k,l))

                        norm = norm + nuclear_charges(i) * nuclear_charges(l) / distance_matrix(i,l) + &
                            & distance_matrix(i,l) * nuclear_charges(l) * nuclear_charges(j) / &
                            & ((distance_matrix(i,l)+distance_matrix(i,j))*distance_matrix(j,l)) + &
                            & distance_matrix(i,l) * nuclear_charges(l) * nuclear_charges(k) / &
                            & ((distance_matrix(i,l)+distance_matrix(i,k))*distance_matrix(k,l))

                        norm = one_twelfth * norm * decay_matrix(i,j) * decay_matrix(i,k) * decay_matrix(i,l) * &
                            & decay_matrix(j,k) * decay_matrix(j,l) * decay_matrix(k,l)

                        four_body(i,j,k,l) = norm
                        four_body(i,j,l,k) = norm
                        four_body(i,k,j,l) = norm
                        four_body(i,k,l,j) = norm
                        four_body(i,l,j,k) = norm
                        four_body(i,l,k,j) = norm
                    enddo
                enddo
            enddo
        enddo
        !$OMP END PARALLEL DO
    else if (potential_type == 4) then
        ! 2: centered on 1, averaged over 2
        ! = R12*Z2*Z3/((R12+R13)*R23) + R12*Z2*Z4/((R12+R14)*R24)
        !$OMP PARALLEL DO PRIVATE(norm) COLLAPSE(2) SCHEDULE(dynamic)
        do i=1, natoms
            do j=1, natoms
                if (i == j) cycle
                if (mask(i,j)) cycle
                do k=j+1, natoms
                    if (i == k) cycle
                    if (mask(i,k) .OR. mask(j,k)) cycle
                    do l=k+1, natoms
                        if (l == i) cycle
                        if (mask(i,l) .OR. mask(j,l) .OR. mask(k,l)) cycle

                        norm = distance_matrix(i,j) * nuclear_charges(j) * nuclear_charges(k) / &
                            & ((distance_matrix(i,j)+distance_matrix(i,k))*distance_matrix(j,k)) + &
                            & distance_matrix(i,j) * nuclear_charges(j) * nuclear_charges(l) / &
                            & ((distance_matrix(i,j)+distance_matrix(i,l))*distance_matrix(j,l))

                        norm = norm + distance_matrix(i,k) * nuclear_charges(k) * nuclear_charges(j) / &
                            & ((distance_matrix(i,k)+distance_matrix(i,j))*distance_matrix(j,k)) + &
                            & distance_matrix(i,k) * nuclear_charges(k) * nuclear_charges(l) / &
                            & ((distance_matrix(i,k)+distance_matrix(i,l))*distance_matrix(k,l))

                        norm = norm + distance_matrix(i,l) * nuclear_charges(l) * nuclear_charges(j) / &
                            & ((distance_matrix(i,l)+distance_matrix(i,j))*distance_matrix(j,l)) + &
                            & distance_matrix(i,l) * nuclear_charges(l) * nuclear_charges(k) / &
                            & ((distance_matrix(i,l)+distance_matrix(i,k))*distance_matrix(k,l))


                        norm = one_twelfth * norm * decay_matrix(i,j) * decay_matrix(i,k) * decay_matrix(i,l) * &
                            & decay_matrix(j,k) * decay_matrix(j,l) * decay_matrix(k,l)

                        four_body(i,j,k,l) = norm
                        four_body(i,j,l,k) = norm
                        four_body(i,k,j,l) = norm
                        four_body(i,k,l,j) = norm
                        four_body(i,l,j,k) = norm
                        four_body(i,l,k,j) = norm
                    enddo
                enddo
            enddo
        enddo
        !$OMP END PARALLEL DO
    else if (potential_type == 5) then
        ! 1: centered on 2, 1 averaged 
        !    Z1*Z2/R12 + R12*Z2*Z3/((R12+R13)*R23) + R12*Z2*Z4/((R12+R14)*R24)
        !$OMP PARALLEL DO PRIVATE(norm) COLLAPSE(2) SCHEDULE(dynamic)
        do i=1, natoms
            do j=1, natoms
                if (i == j) cycle
                if (mask(i,j)) cycle
                do k=i+1, natoms
                    if (j == k) cycle
                    if (mask(i,k) .OR. mask(j,k)) cycle
                    do l=k+1, natoms
                        if (l == j) cycle
                        if (mask(i,l) .OR. mask(j,l) .OR. mask(k,l)) cycle

                        norm = nuclear_charges(i) * nuclear_charges(j) / distance_matrix(i,j) + &
                            & distance_matrix(i,j) * nuclear_charges(j) * nuclear_charges(k) / &
                            & ((distance_matrix(i,j)+distance_matrix(i,k))*distance_matrix(j,k)) + &
                            & distance_matrix(i,j) * nuclear_charges(j) * nuclear_charges(l) / &
                            & ((distance_matrix(i,j)+distance_matrix(i,l))*distance_matrix(j,l))

                        norm = norm + nuclear_charges(k) * nuclear_charges(j) / distance_matrix(k,j) + &
                            & distance_matrix(k,j) * nuclear_charges(j) * nuclear_charges(i) / &
                            & ((distance_matrix(k,j)+distance_matrix(i,k))*distance_matrix(j,i)) + &
                            & distance_matrix(k,j) * nuclear_charges(j) * nuclear_charges(l) / &
                            & ((distance_matrix(k,j)+distance_matrix(k,l))*distance_matrix(j,l))

                        norm = norm + nuclear_charges(l) * nuclear_charges(j) / distance_matrix(l,j) + &
                            & distance_matrix(l,j) * nuclear_charges(j) * nuclear_charges(k) / &
                            & ((distance_matrix(l,j)+distance_matrix(l,k))*distance_matrix(j,k)) + &
                            & distance_matrix(l,j) * nuclear_charges(j) * nuclear_charges(i) / &
                            & ((distance_matrix(l,j)+distance_matrix(i,l))*distance_matrix(j,i))

                        norm = one_twelfth * norm * decay_matrix(i,j) * decay_matrix(i,k) * decay_matrix(i,l) * &
                            & decay_matrix(j,k) * decay_matrix(j,l) * decay_matrix(k,l)

                        four_body(i,j,k,l) = norm
                        four_body(i,j,l,k) = norm
                        four_body(k,j,i,l) = norm
                        four_body(k,j,l,i) = norm
                        four_body(l,j,k,i) = norm
                        four_body(l,j,i,k) = norm
                    enddo
                enddo
            enddo
        enddo
        !$OMP END PARALLEL DO
    else if (potential_type == 6) then
        ! 1: centered on 2, 1 averaged 
        !    R12*Z2*Z3/((R12+R13)*R23) + R12*Z2*Z4/((R12+R14)*R24)
        !$OMP PARALLEL DO PRIVATE(norm) COLLAPSE(2) SCHEDULE(dynamic)
        do i=1, natoms
            do j=1, natoms
                if (i == j) cycle
                if (mask(i,j)) cycle
                do k=i+1, natoms
                    if (j == k) cycle
                    if (mask(i,k) .OR. mask(j,k)) cycle
                    do l=k+1, natoms
                        if (l == j) cycle
                        if (mask(i,l) .OR. mask(j,l) .OR. mask(k,l)) cycle

                        norm = distance_matrix(i,j) * nuclear_charges(j) * nuclear_charges(k) / &
                            & ((distance_matrix(i,j)+distance_matrix(i,k))*distance_matrix(j,k)) + &
                            & distance_matrix(i,j) * nuclear_charges(j) * nuclear_charges(l) / &
                            & ((distance_matrix(i,j)+distance_matrix(i,l))*distance_matrix(j,l))

                        norm = norm + distance_matrix(k,j) * nuclear_charges(j) * nuclear_charges(i) / &
                            & ((distance_matrix(k,j)+distance_matrix(i,k))*distance_matrix(j,i)) + &
                            & distance_matrix(k,j) * nuclear_charges(j) * nuclear_charges(l) / &
                            & ((distance_matrix(k,j)+distance_matrix(k,l))*distance_matrix(j,l))

                        norm = norm + distance_matrix(l,j) * nuclear_charges(j) * nuclear_charges(k) / &
                            & ((distance_matrix(l,j)+distance_matrix(l,k))*distance_matrix(j,k)) + &
                            & distance_matrix(l,j) * nuclear_charges(j) * nuclear_charges(i) / &
                            & ((distance_matrix(l,j)+distance_matrix(i,l))*distance_matrix(j,i))

                        norm = one_twelfth * norm * decay_matrix(i,j) * decay_matrix(i,k) * decay_matrix(i,l) * &
                            & decay_matrix(j,k) * decay_matrix(j,l) * decay_matrix(k,l)

                        four_body(i,j,k,l) = norm
                        four_body(i,j,l,k) = norm
                        four_body(k,j,i,l) = norm
                        four_body(k,j,l,i) = norm
                        four_body(l,j,k,i) = norm
                        four_body(l,j,i,k) = norm
                    enddo
                enddo
            enddo
        enddo
        !$OMP END PARALLEL DO
    else if (potential_type == 7) then
        ! 1: 1,2 averaged
        !    Z1*Z2/R12 + R12*Z2*Z3/((R12+R13)*R23) + R12*Z2*Z4/((R12+R14)*R24)
        !$OMP PARALLEL DO PRIVATE(norm)  SCHEDULE(dynamic)
        do i=1, natoms
            do j=i+1, natoms
                if (mask(i,j)) cycle
                do k=j+1, natoms
                    if (mask(i,k) .OR. mask(j,k)) cycle
                    do l=k+1, natoms
                        if (mask(i,l) .OR. mask(j,l) .OR. mask(k,l)) cycle

                        norm = nuclear_charges(i) * nuclear_charges(j) / distance_matrix(i,j) + &
                            & distance_matrix(i,j) * nuclear_charges(j) * nuclear_charges(k) / &
                            & ((distance_matrix(i,j)+distance_matrix(i,k))*distance_matrix(j,k)) + &
                            & distance_matrix(i,j) * nuclear_charges(j) * nuclear_charges(l) / &
                            & ((distance_matrix(i,j)+distance_matrix(i,l))*distance_matrix(j,l))

                        norm = norm + nuclear_charges(i) * nuclear_charges(j) / distance_matrix(i,j) + &
                            & distance_matrix(i,j) * nuclear_charges(i) * nuclear_charges(k) / &
                            & ((distance_matrix(i,j)+distance_matrix(j,k))*distance_matrix(i,k)) + &
                            & distance_matrix(i,j) * nuclear_charges(i) * nuclear_charges(l) / &
                            & ((distance_matrix(i,j)+distance_matrix(j,l))*distance_matrix(i,l))

                        norm = norm + nuclear_charges(k) * nuclear_charges(j) / distance_matrix(k,j) + &
                            & distance_matrix(k,j) * nuclear_charges(k) * nuclear_charges(i) / &
                            & ((distance_matrix(k,j)+distance_matrix(j,i))*distance_matrix(i,k)) + &
                            & distance_matrix(k,j) * nuclear_charges(k) * nuclear_charges(l) / &
                            & ((distance_matrix(k,j)+distance_matrix(j,l))*distance_matrix(k,l))

                        norm = norm + nuclear_charges(k) * nuclear_charges(j) / distance_matrix(k,j) + &
                            & distance_matrix(k,j) * nuclear_charges(j) * nuclear_charges(i) / &
                            & ((distance_matrix(k,j)+distance_matrix(k,i))*distance_matrix(i,j)) + &
                            & distance_matrix(k,j) * nuclear_charges(j) * nuclear_charges(l) / &
                            & ((distance_matrix(k,j)+distance_matrix(k,l))*distance_matrix(j,l))

                        norm = norm + nuclear_charges(l) * nuclear_charges(j) / distance_matrix(l,j) + &
                            & distance_matrix(l,j) * nuclear_charges(l) * nuclear_charges(k) / &
                            & ((distance_matrix(l,j)+distance_matrix(j,k))*distance_matrix(l,k)) + &
                            & distance_matrix(l,j) * nuclear_charges(i) * nuclear_charges(l) / &
                            & ((distance_matrix(l,j)+distance_matrix(j,i))*distance_matrix(i,l))

                        norm = norm + nuclear_charges(l) * nuclear_charges(j) / distance_matrix(l,j) + &
                            & distance_matrix(l,j) * nuclear_charges(j) * nuclear_charges(k) / &
                            & ((distance_matrix(l,j)+distance_matrix(l,k))*distance_matrix(j,k)) + &
                            & distance_matrix(l,j) * nuclear_charges(i) * nuclear_charges(j) / &
                            & ((distance_matrix(l,j)+distance_matrix(l,i))*distance_matrix(i,j))

                        norm = norm + nuclear_charges(i) * nuclear_charges(k) / distance_matrix(i,k) + &
                            & distance_matrix(i,k) * nuclear_charges(i) * nuclear_charges(j) / &
                            & ((distance_matrix(i,k)+distance_matrix(j,k))*distance_matrix(i,j)) + &
                            & distance_matrix(i,k) * nuclear_charges(i) * nuclear_charges(l) / &
                            & ((distance_matrix(i,k)+distance_matrix(k,l))*distance_matrix(i,l))

                        norm = norm + nuclear_charges(i) * nuclear_charges(k) / distance_matrix(i,k) + &
                            & distance_matrix(i,k) * nuclear_charges(k) * nuclear_charges(j) / &
                            & ((distance_matrix(i,k)+distance_matrix(j,i))*distance_matrix(k,j)) + &
                            & distance_matrix(i,k) * nuclear_charges(k) * nuclear_charges(l) / &
                            & ((distance_matrix(i,k)+distance_matrix(i,l))*distance_matrix(k,l))

                        norm = norm + nuclear_charges(i) * nuclear_charges(l) / distance_matrix(i,l) + &
                            & distance_matrix(i,l) * nuclear_charges(i) * nuclear_charges(k) / &
                            & ((distance_matrix(i,l)+distance_matrix(l,k))*distance_matrix(i,k)) + &
                            & distance_matrix(i,l) * nuclear_charges(i) * nuclear_charges(j) / &
                            & ((distance_matrix(i,l)+distance_matrix(j,l))*distance_matrix(i,j))

                        norm = norm + nuclear_charges(i) * nuclear_charges(l) / distance_matrix(i,l) + &
                            & distance_matrix(i,l) * nuclear_charges(l) * nuclear_charges(k) / &
                            & ((distance_matrix(i,l)+distance_matrix(i,k))*distance_matrix(l,k)) + &
                            & distance_matrix(i,l) * nuclear_charges(l) * nuclear_charges(j) / &
                            & ((distance_matrix(i,l)+distance_matrix(j,i))*distance_matrix(l,j))

                        norm = norm + nuclear_charges(l) * nuclear_charges(k) / distance_matrix(l,k) + &
                            & distance_matrix(l,k) * nuclear_charges(l) * nuclear_charges(j) / &
                            & ((distance_matrix(l,k)+distance_matrix(j,k))*distance_matrix(l,j)) + &
                            & distance_matrix(l,k) * nuclear_charges(i) * nuclear_charges(l) / &
                            & ((distance_matrix(l,k)+distance_matrix(k,i))*distance_matrix(i,l))

                        norm = norm + nuclear_charges(l) * nuclear_charges(k) / distance_matrix(l,k) + &
                            & distance_matrix(l,k) * nuclear_charges(k) * nuclear_charges(j) / &
                            & ((distance_matrix(l,k)+distance_matrix(j,l))*distance_matrix(k,j)) + &
                            & distance_matrix(l,k) * nuclear_charges(i) * nuclear_charges(k) / &
                            & ((distance_matrix(l,k)+distance_matrix(l,i))*distance_matrix(i,k))

                        norm = one_twelfth * norm * decay_matrix(i,j) * decay_matrix(i,k) * decay_matrix(i,l) * &
                            & decay_matrix(j,k) * decay_matrix(j,l) * decay_matrix(k,l)

                        four_body(i,j,k,l) = norm
                        four_body(i,j,l,k) = norm
                        four_body(j,i,k,l) = norm
                        four_body(j,i,l,k) = norm
                        four_body(k,j,i,l) = norm
                        four_body(k,j,l,i) = norm
                        four_body(j,k,i,l) = norm
                        four_body(j,k,l,i) = norm
                        four_body(l,j,k,i) = norm
                        four_body(l,j,i,k) = norm
                        four_body(j,l,k,i) = norm
                        four_body(j,l,i,k) = norm
                        four_body(i,k,j,l) = norm
                        four_body(i,k,l,j) = norm
                        four_body(k,i,j,l) = norm
                        four_body(k,i,l,j) = norm
                        four_body(i,l,k,j) = norm
                        four_body(i,l,j,k) = norm
                        four_body(l,i,k,j) = norm
                        four_body(l,i,j,k) = norm
                        four_body(k,l,i,j) = norm
                        four_body(k,l,j,i) = norm
                        four_body(l,k,i,j) = norm
                        four_body(l,k,j,i) = norm
                    enddo
                enddo
            enddo
        enddo
        !$OMP END PARALLEL DO
    else if (potential_type == 8) then
        ! 1: 1,2 averaged
        !    R12*Z2*Z3/((R12+R13)*R23) + R12*Z2*Z4/((R12+R14)*R24)
        !$OMP PARALLEL DO PRIVATE(norm)  SCHEDULE(dynamic)
        do i=1, natoms
            do j=i+1, natoms
                if (mask(i,j)) cycle
                do k=j+1, natoms
                    if (mask(i,k) .OR. mask(j,k)) cycle
                    do l=k+1, natoms
                        if (mask(i,l) .OR. mask(j,l) .OR. mask(k,l)) cycle

                        norm = distance_matrix(i,j) * nuclear_charges(j) * nuclear_charges(k) / &
                            & ((distance_matrix(i,j)+distance_matrix(i,k))*distance_matrix(j,k)) + &
                            & distance_matrix(i,j) * nuclear_charges(j) * nuclear_charges(l) / &
                            & ((distance_matrix(i,j)+distance_matrix(i,l))*distance_matrix(j,l))

                        norm = norm + distance_matrix(i,j) * nuclear_charges(i) * nuclear_charges(k) / &
                            & ((distance_matrix(i,j)+distance_matrix(j,k))*distance_matrix(i,k)) + &
                            & distance_matrix(i,j) * nuclear_charges(i) * nuclear_charges(l) / &
                            & ((distance_matrix(i,j)+distance_matrix(j,l))*distance_matrix(i,l))

                        norm = norm + distance_matrix(k,j) * nuclear_charges(k) * nuclear_charges(i) / &
                            & ((distance_matrix(k,j)+distance_matrix(j,i))*distance_matrix(i,k)) + &
                            & distance_matrix(k,j) * nuclear_charges(k) * nuclear_charges(l) / &
                            & ((distance_matrix(k,j)+distance_matrix(j,l))*distance_matrix(k,l))

                        norm = norm + distance_matrix(k,j) * nuclear_charges(j) * nuclear_charges(i) / &
                            & ((distance_matrix(k,j)+distance_matrix(k,i))*distance_matrix(i,j)) + &
                            & distance_matrix(k,j) * nuclear_charges(j) * nuclear_charges(l) / &
                            & ((distance_matrix(k,j)+distance_matrix(k,l))*distance_matrix(j,l))

                        norm = norm + distance_matrix(l,j) * nuclear_charges(l) * nuclear_charges(k) / &
                            & ((distance_matrix(l,j)+distance_matrix(j,k))*distance_matrix(l,k)) + &
                            & distance_matrix(l,j) * nuclear_charges(i) * nuclear_charges(l) / &
                            & ((distance_matrix(l,j)+distance_matrix(j,i))*distance_matrix(i,l))

                        norm = norm + distance_matrix(l,j) * nuclear_charges(j) * nuclear_charges(k) / &
                            & ((distance_matrix(l,j)+distance_matrix(l,k))*distance_matrix(j,k)) + &
                            & distance_matrix(l,j) * nuclear_charges(i) * nuclear_charges(j) / &
                            & ((distance_matrix(l,j)+distance_matrix(l,i))*distance_matrix(i,j))

                        norm = norm + distance_matrix(i,k) * nuclear_charges(i) * nuclear_charges(j) / &
                            & ((distance_matrix(i,k)+distance_matrix(j,k))*distance_matrix(i,j)) + &
                            & distance_matrix(i,k) * nuclear_charges(i) * nuclear_charges(l) / &
                            & ((distance_matrix(i,k)+distance_matrix(k,l))*distance_matrix(i,l))

                        norm = norm + distance_matrix(i,k) * nuclear_charges(k) * nuclear_charges(j) / &
                            & ((distance_matrix(i,k)+distance_matrix(j,i))*distance_matrix(k,j)) + &
                            & distance_matrix(i,k) * nuclear_charges(k) * nuclear_charges(l) / &
                            & ((distance_matrix(i,k)+distance_matrix(i,l))*distance_matrix(k,l))

                        norm = norm + distance_matrix(i,l) * nuclear_charges(i) * nuclear_charges(k) / &
                            & ((distance_matrix(i,l)+distance_matrix(l,k))*distance_matrix(i,k)) + &
                            & distance_matrix(i,l) * nuclear_charges(i) * nuclear_charges(j) / &
                            & ((distance_matrix(i,l)+distance_matrix(j,l))*distance_matrix(i,j))

                        norm = norm + distance_matrix(i,l) * nuclear_charges(l) * nuclear_charges(k) / &
                            & ((distance_matrix(i,l)+distance_matrix(i,k))*distance_matrix(l,k)) + &
                            & distance_matrix(i,l) * nuclear_charges(l) * nuclear_charges(j) / &
                            & ((distance_matrix(i,l)+distance_matrix(j,i))*distance_matrix(l,j))

                        norm = norm + distance_matrix(l,k) * nuclear_charges(l) * nuclear_charges(j) / &
                            & ((distance_matrix(l,k)+distance_matrix(j,k))*distance_matrix(l,j)) + &
                            & distance_matrix(l,k) * nuclear_charges(i) * nuclear_charges(l) / &
                            & ((distance_matrix(l,k)+distance_matrix(k,i))*distance_matrix(i,l))

                        norm = norm + distance_matrix(l,k) * nuclear_charges(k) * nuclear_charges(j) / &
                            & ((distance_matrix(l,k)+distance_matrix(j,l))*distance_matrix(k,j)) + &
                            & distance_matrix(l,k) * nuclear_charges(i) * nuclear_charges(k) / &
                            & ((distance_matrix(l,k)+distance_matrix(l,i))*distance_matrix(i,k))

                        norm = one_twelfth * norm * decay_matrix(i,j) * decay_matrix(i,k) * decay_matrix(i,l) * &
                            & decay_matrix(j,k) * decay_matrix(j,l) * decay_matrix(k,l)

                        four_body(i,j,k,l) = norm
                        four_body(i,j,l,k) = norm
                        four_body(j,i,k,l) = norm
                        four_body(j,i,l,k) = norm
                        four_body(k,j,i,l) = norm
                        four_body(k,j,l,i) = norm
                        four_body(j,k,i,l) = norm
                        four_body(j,k,l,i) = norm
                        four_body(l,j,k,i) = norm
                        four_body(l,j,i,k) = norm
                        four_body(j,l,k,i) = norm
                        four_body(j,l,i,k) = norm
                        four_body(i,k,j,l) = norm
                        four_body(i,k,l,j) = norm
                        four_body(k,i,j,l) = norm
                        four_body(k,i,l,j) = norm
                        four_body(i,l,k,j) = norm
                        four_body(i,l,j,k) = norm
                        four_body(l,i,k,j) = norm
                        four_body(l,i,j,k) = norm
                        four_body(k,l,i,j) = norm
                        four_body(k,l,j,i) = norm
                        four_body(l,k,i,j) = norm
                        four_body(l,k,j,i) = norm
                    enddo
                enddo
            enddo
        enddo
        !$OMP END PARALLEL DO
    endif

end subroutine cm_global_four_body


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
    allocate(distance_matrix_tmp(natoms, natoms))
    allocate(sorted_atoms_all(natoms, central_natoms))

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

    logical, allocatable, dimension(:, :) :: mask


    double precision, allocatable, dimension(:) :: bag
    double precision, allocatable, dimension(:, :) :: distance_matrix
    double precision, allocatable, dimension(:, :) :: cent_decay_matrix
    double precision, allocatable, dimension(:, :) :: int_decay_matrix

    integer :: i, j, k, l, m, n, nid, type0, type1, type2, nbag, max_idx

    double precision, parameter :: pi = 4.0d0 * atan(1.0d0)


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

    ! Allocate temporary
    allocate(distance_matrix(natoms,natoms))

    huge_double = huge(distance_matrix(1,1))

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
    allocate(index_types(natoms))

    index_types = 1

    ! Get element types
    if (nid > 2) then
        do i = 1, natoms
            type1 = nuclear_charges(i)
            do j = 2, nid
                if (type1 == id(j)) then
                    index_types(i) = j
                    exit
                endif
            enddo
        enddo
    endif

    ! Allocate temporary
    allocate(n_pair_indices(natoms, nid))
    allocate(pair_indices(natoms, nid, natoms))
    allocate(mask(natoms, natoms))
    allocate(cent_decay_matrix(natoms,natoms))
    allocate(int_decay_matrix(natoms,natoms))

    n_pair_indices = 0
    mask = .false.
    cent_decay_matrix = 1.0d0
    int_decay_matrix = 1.0d0

    ! Create convenience arrays
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
                    n_pair_indices(i, type2) = n_pair_indices(i, type2) + 1
                    pair_indices(i, type2, n_pair_indices(i, type2)) = j

                    n_pair_indices(j, type1) = n_pair_indices(j, type1) + 1
                    pair_indices(j, type1, n_pair_indices(j, type1)) = i

                    ! Calculate decay factors
                    decay_dist = norm - cent_cutoff + cent_decay
                    if (decay_dist > 0) then
                        prefactor = 0.5d0 * &
                            & (cos(pi * decay_dist / cent_decay) + 1)

                        cent_decay_matrix(i,j) = cent_decay_matrix(i,j) * prefactor
                        cent_decay_matrix(j,i) = cent_decay_matrix(j,i) * prefactor

                    endif
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
                        int_decay_matrix(j,i) = int_decay_matrix(i,j)
                    endif
                endif
            enddo
        enddo

    ! Check given array sizes
    do j = 1, nid
        n = maxval(n_pair_indices(:,j), dim=1)
        if (n > nmax(j)) then
            write(*,*) "ERROR: Bag of Bonds generation"
            write(*,*) nmax(j), "size set for atom with nuclear charge,", &
                & id(j), "but", n, "size needed!"
        endif
    enddo


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

    ! AB (including AA)
    do i = 1, nid
        do j = i, nid
            if (j == 1) then
                start_indices(1,1,3) = start_indices(nid,1,2) + nmax(nid)
            else if (j == i) then
                start_indices(i,i,3) = start_indices(i-1,nid,3) + nmax(i-1)*nmax(nid)
            else if (j == i+1) then
                start_indices(i,j,3) = start_indices(i,i,3) + (nmax(i)* (nmax(i) - 1)) / 2
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

    ! X bag
    do k = 1, natoms
        cm(k,1) = 0.5d0 * atomic_charges(k) ** 2.4d0
    enddo


    !$OMP PARALLEL DO PRIVATE(type0, nbag, l, norm, pair_norm, bag, n, m) COLLAPSE(2)
        do k = 1, natoms
            do type1 = 1, nid
                type0 = index_types(k)

                ! start A bag
                nbag = n_pair_indices(k, type1)
                do i = 1, nbag
                    l = pair_indices(k, type1, i)

                    norm = distance_matrix(l, k)
                    pair_norm = 0.5d0 * atomic_charges(l) ** 2.4d0 &
                        & * cent_decay_matrix(k,l) ** 2
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
                ! end A bag

                ! start XA bag
                nbag = 0
                do i = 1, n_pair_indices(k,type1)
                    l = pair_indices(k, type1, i)
                    if (mask(l,k)) cycle
                    nbag = nbag + 1

                    norm = distance_matrix(l, k)

                    ! Alternatively, don't include int_decay
                    pair_norm = atomic_charges(k) * atomic_charges(l) / norm &
                        & * cent_decay_matrix(k,l) &
                        & * int_decay_matrix(k,l)

                    bag(nbag) = pair_norm
                enddo

                ! sort
                n = start_indices(type1, 1, 2)
                do i = 1, nbag
                    l = maxloc(bag(:nbag), dim=1)
                    cm(k, n + i) = bag(l)
                    bag(l) = -huge_double
                enddo
                ! end sort
                ! end XA bag

                ! start AA bag
                nbag = 0
                do i = 1, n_pair_indices(k, type1)
                    l = pair_indices(k, type1, i)
                    do j = 1, i-1
                        m = pair_indices(k, type1, j)
                        if (mask(l,m)) cycle
                        nbag = nbag + 1
                        norm = distance_matrix(l, m)

                        pair_norm = atomic_charges(l) * atomic_charges(m) / norm &
                            & * cent_decay_matrix(k,l) * cent_decay_matrix(k, m) &
                            & * int_decay_matrix(l,m)

                        bag(nbag) = pair_norm
                    enddo
                enddo

                ! sort
                n = start_indices(type1, type1, 3)
                do i = 1, nbag
                    l = maxloc(bag(:nbag), dim=1)
                    cm(k, n + i) = bag(l)
                    bag(l) = -huge_double
                enddo
                ! end sort
                ! end AA bag

                ! start AB bag
                do type2 = type1+1, nid
                    nbag = 0
                    do i = 1, n_pair_indices(k, type1)
                        l = pair_indices(k, type1, i)
                        do j = 1, n_pair_indices(k, type2)
                            m = pair_indices(k, type2, j)
                            if (mask(l,m)) cycle
                            nbag = nbag + 1
                            norm = distance_matrix(l, m)

                            pair_norm = atomic_charges(l) * atomic_charges(m) / norm &
                                & * cent_decay_matrix(k,l) * cent_decay_matrix(k, m) &
                                & * int_decay_matrix(l,m)


                            bag(nbag) = pair_norm
                        enddo
                    enddo

                    ! sort
                    n = start_indices(type1, type2, 3)
                    do i = 1, nbag
                        l = maxloc(bag(:nbag), dim=1)
                        cm(k, n + i) = bag(l)
                        bag(l) = -huge_double
                    enddo
                    ! end sort
                    ! end AB bag
                enddo
            enddo
        enddo
    !$OMP END PARALLEL DO

    ! Clean up
    deallocate(index_types)
    deallocate(n_pair_indices)
    deallocate(pair_indices)
    deallocate(mask)
    deallocate(distance_matrix)
    deallocate(cent_decay_matrix)
    deallocate(int_decay_matrix)
    deallocate(bag)
    deallocate(start_indices)

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

subroutine fgenerate_local_bob_sncf(atomic_charges, coordinates, nuclear_charges, id, &
        & nmax, natoms, cent_cutoff, cent_decay, int_cutoff, int_decay, localization, alt, ncm, cm)


    implicit none

    double precision, dimension(:), intent(in) :: atomic_charges
    double precision, dimension(:,:), intent(in) :: coordinates
    integer, dimension(:), intent(in) :: nuclear_charges
    integer, dimension(:), intent(in) :: id
    integer, dimension(:), intent(in) :: nmax
    integer, intent(in) :: natoms
    double precision, intent(inout) :: cent_cutoff, cent_decay, int_cutoff, int_decay
    integer, intent(in) :: localization
    logical, intent(in) :: alt
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

    logical, allocatable, dimension(:, :) :: mask


    double precision, allocatable, dimension(:) :: bag
    double precision, allocatable, dimension(:, :) :: distance_matrix
    double precision, allocatable, dimension(:, :) :: cent_decay_matrix
    double precision, allocatable, dimension(:, :) :: int_decay_matrix

    integer :: i, j, k, l, m, n, nid, type0, type1, type2, nbag, max_idx

    double precision, parameter :: pi = 4.0d0 * atan(1.0d0)


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

    ! Allocate temporary
    allocate(distance_matrix(natoms,natoms))

    huge_double = huge(distance_matrix(1,1))

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
    allocate(index_types(natoms))

    index_types = 1

    ! Get element types
    if (nid > 2) then
        do i = 1, natoms
            type1 = nuclear_charges(i)
            do j = 2, nid
                if (type1 == id(j)) then
                    index_types(i) = j
                    exit
                endif
            enddo
        enddo
    endif

    ! Allocate temporary
    allocate(n_pair_indices(natoms, nid))
    allocate(pair_indices(natoms, nid, natoms))
    allocate(mask(natoms, natoms))
    allocate(cent_decay_matrix(natoms,natoms))
    allocate(int_decay_matrix(natoms,natoms))

    n_pair_indices = 0
    mask = .false.
    cent_decay_matrix = 1.0d0
    int_decay_matrix = 1.0d0

    ! Create convenience arrays
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
                    n_pair_indices(i, type2) = n_pair_indices(i, type2) + 1
                    pair_indices(i, type2, n_pair_indices(i, type2)) = j

                    n_pair_indices(j, type1) = n_pair_indices(j, type1) + 1
                    pair_indices(j, type1, n_pair_indices(j, type1)) = i

                    ! Calculate decay factors
                    decay_dist = norm - cent_cutoff + cent_decay
                    if (decay_dist > 0) then
                        prefactor = 0.5d0 * &
                            & (cos(pi * decay_dist / cent_decay) + 1)

                        cent_decay_matrix(i,j) = cent_decay_matrix(i,j) * prefactor
                        cent_decay_matrix(j,i) = cent_decay_matrix(j,i) * prefactor

                    endif
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
                        int_decay_matrix(j,i) = int_decay_matrix(i,j)
                    endif
                endif
            enddo
        enddo

    ! Check given array sizes
    do j = 1, nid
        n = maxval(n_pair_indices(:,j), dim=1)
        if (n > nmax(j)) then
            write(*,*) "ERROR: Bag of Bonds generation"
            write(*,*) nmax(j), "size set for atom with nuclear charge,", &
                & id(j), "but", n, "size needed!"
        endif
    enddo


    ! Allocate temporary
    allocate(start_indices(nid,nid,3))

    ! get start indices
    ! bags with atom types A,B,C with k of type X in (A,B,C):
    ! (X, XA, XB, XC, AA, AB, AC, BB, BC, CC)
    ! XA
    do i = 1, nid
        if (i == 1) then
            start_indices(1,1,1) = 1
        else
            start_indices(i,1,1) = start_indices(i-1,1,1) + nmax(i-1)
        endif
    enddo

    ! AB (including AA)
    do i = 1, nid
        do j = i, nid
            if (j == 1) then
                start_indices(1,1,2) = start_indices(nid,1,1) + nmax(nid)
            else if (j == i) then
                start_indices(i,i,2) = start_indices(i-1,nid,2) + nmax(i-1)*nmax(nid)
            else if (j == i+1) then
                start_indices(i,j,2) = start_indices(i,i,2) + (nmax(i)* (nmax(i) + 1)) / 2
            else
                start_indices(i,j,2) = start_indices(i,j-1,2) + nmax(i)*nmax(j-1)
            endif
        enddo
    enddo


    ! Get max bag size
    nbag = 1
    do i = 1, nid
        nbag = max(nbag, nmax(i))
        nbag = max(nbag, (nmax(i)*(nmax(i)+1))/2)
        do j = i + 1, nid
            nbag = max(nbag, nmax(i)*nmax(j))
        enddo
    enddo

    ! Allocate temporary
    allocate(bag(nbag))

    ! Construct representation
    cm = 0.0d0

    ! X bag
    do k = 1, natoms
        cm(k,1) = 0.5d0 * atomic_charges(k) ** 2.4d0
    enddo


    !$OMP PARALLEL DO PRIVATE(type0, nbag, l, norm, pair_norm, bag, n, m) COLLAPSE(2)
        do k = 1, natoms
            do type1 = 1, nid
                type0 = index_types(k)

                ! start XA bag
                nbag = 0
                do i = 1, n_pair_indices(k,type1)
                    l = pair_indices(k, type1, i)
                    if (mask(l,k)) cycle
                    nbag = nbag + 1

                    norm = distance_matrix(l, k)
                    if (alt) then
                        norm = 2*norm
                    endif

                    ! Alternatively, don't include int_decay
                    pair_norm = atomic_charges(k) * atomic_charges(l) / norm**localization &
                        & * cent_decay_matrix(k,l) &
                        & * int_decay_matrix(k,l)

                    bag(nbag) = pair_norm
                enddo

                ! sort
                n = start_indices(type1, 1, 1)
                do i = 1, nbag
                    l = maxloc(bag(:nbag), dim=1)
                    cm(k, n + i) = bag(l)
                    bag(l) = -huge_double
                enddo
                ! end sort
                ! end XA bag

                ! start AA bag
                nbag = 0
                do i = 1, n_pair_indices(k, type1)
                    l = pair_indices(k, type1, i)
                    do j = 1, i
                        m = pair_indices(k, type1, j)
                        if (mask(l,m)) cycle
                        nbag = nbag + 1
                        norm = distance_matrix(l,k) + distance_matrix(m,k)
                        if (alt) then
                            norm = norm + distance_matrix(l, m)
                        endif

                        pair_norm = atomic_charges(l) * atomic_charges(m) / norm**localization &
                            & * cent_decay_matrix(k,l) * cent_decay_matrix(k, m) &
                            & * int_decay_matrix(l,m)

                        bag(nbag) = pair_norm
                    enddo
                enddo

                ! sort
                n = start_indices(type1, type1, 2)
                do i = 1, nbag
                    l = maxloc(bag(:nbag), dim=1)
                    cm(k, n + i) = bag(l)
                    bag(l) = -huge_double
                enddo
                ! end sort
                ! end AA bag

                ! start AB bag
                do type2 = type1+1, nid
                    nbag = 0
                    do i = 1, n_pair_indices(k, type1)
                        l = pair_indices(k, type1, i)
                        do j = 1, n_pair_indices(k, type2)
                            m = pair_indices(k, type2, j)
                            if (mask(l,m)) cycle
                            nbag = nbag + 1
                            norm = distance_matrix(l,k) + distance_matrix(m,k)
                            if (alt) then
                                norm = norm + distance_matrix(l, m)
                            endif

                            pair_norm = atomic_charges(l) * atomic_charges(m) / norm**localization &
                                & * cent_decay_matrix(k,l) * cent_decay_matrix(k, m) &
                                & * int_decay_matrix(l,m)


                            bag(nbag) = pair_norm
                        enddo
                    enddo

                    ! sort
                    n = start_indices(type1, type2, 2)
                    do i = 1, nbag
                        l = maxloc(bag(:nbag), dim=1)
                        cm(k, n + i) = bag(l)
                        bag(l) = -huge_double
                    enddo
                    ! end sort
                    ! end AB bag
                enddo
            enddo
        enddo
    !$OMP END PARALLEL DO

    ! Clean up
    deallocate(index_types)
    deallocate(n_pair_indices)
    deallocate(pair_indices)
    deallocate(mask)
    deallocate(distance_matrix)
    deallocate(cent_decay_matrix)
    deallocate(int_decay_matrix)
    deallocate(bag)
    deallocate(start_indices)

end subroutine fgenerate_local_bob_sncf

subroutine fgenerate_global(nuclear_charges, coordinates, potential_types, sorting, charge_types, &
        & nsize, nasize, cutoff, decay, localization, weights_one, weights_two, weight_three, weight_four, &
        & nrep, representation)

    use representations, only: global_checks, get_distance_matrix,get_index_types, &
        & get_convenience_arrays_global, smoothen_distance_matrix, &
        & cm_global_three_body, cm_global_four_body
    implicit none

    ! Nuclear charges (size = N)
    integer, dimension(:), intent(inout) :: nuclear_charges
    ! Coordinates (size = N,3)
    double precision, dimension(:,:), intent(inout) :: coordinates
    ! potential types (size = 5)
        ! index1: self interaction
        ! 0: None
        ! 1: 0.5 * Z**2.4
        ! index2: two body
        ! 0: None
        ! 1: Z1*Z2/R12
        ! Index3: Three body
        ! 0: None
        ! 1: centered on 1: Z2*Z3 / (R12 + R13)
        ! 2: centered on 1: Z2*Z3 / (R12 + R13 + R23)
        ! 3: centered on 1: Z1*(Z2+Z3)/(r13 - m2/(m2+m3) * r23)
        ! 4: centered on 1: Z2*Z3/R23 + Z1*(Z2+Z3)/(r13 - m2/(m2+m3) * r23)
        ! 5: (1) average
        ! 6: (2) average
        ! 7: (3) average
        ! 8: (4) average
        ! 9: (Z1*Z2*Z3/(Z1+Z2+Z3)) * (3 * cos(THETA123) * cos(THETA312) * cos(THETA231) + 1) / (R12*R13*R23)**3
        ! Index4: Four body
        ! 0: None
        ! 1: centered on 1,2: = Z1*Z2/R12 + R12*Z2*Z3/((R12+R13)*R23) + R12*Z2*Z4/((R12+R14)*R24)
        ! 2: centered on 1,2: = R12*Z2*Z3/((R12+R13)*R23) + R12*Z2*Z4/((R12+R14)*R24)
        ! 3: (1) average over second atom
        ! 4: (2) average over second atom
        ! 5: (1) average over first atom
        ! 6: (2) average over first atom
        ! 7: (1) averaged over all atoms
        ! 8: (2) averaged over all atoms
        ! index5: erf approxmation
        ! 0: no
        ! 1: yes
    integer, dimension(:), intent(in) :: potential_types
    ! Sorting
        ! 1: unsorted
        ! 2: distance from center of mass
        ! 3: 2-norm
        ! 4: bags
    integer, intent(in) :: sorting
    ! nuclear charge of the different atom types
    integer, dimension(:), intent(in) :: charge_types
    ! maximum size of arrays
    integer, intent(in) :: nsize
    integer, dimension(:), intent(in) :: nasize
    ! cutoff parameters
    double precision, intent(in) :: cutoff
    double precision, intent(in) :: decay
    ! localization for two body
    integer, intent(in) :: localization
    ! weights
    double precision, dimension(:), intent(in) :: weights_one
    double precision, dimension(:,:), intent(in) :: weights_two
    double precision, intent(in) :: weight_three
    double precision, intent(in) :: weight_four
    ! size of representation
    integer, intent(in) :: nrep
    ! representation
    double precision, dimension(nrep), intent(out) :: representation

    ! temporary
    integer :: i,j, m,n, a,b, k,l, nbag, idx, idx0, idx1, type1, type2, type3, type4, natypes, natoms
    double precision :: norm
    double precision :: huge_double
    double precision, allocatable, dimension(:) :: center_of_mass
    double precision, allocatable, dimension(:) :: row_norm
    double precision, allocatable, dimension(:) :: one_body
    double precision, allocatable, dimension(:,:) :: distance_matrix
    double precision, allocatable, dimension(:,:) :: two_body
    double precision, allocatable, dimension(:, :) :: decay_matrix
    double precision, allocatable, dimension(:,:,:) :: three_body
    double precision, allocatable, dimension(:,:,:,:) :: four_body
    integer, allocatable, dimension(:)  :: sorted_atoms
    integer, allocatable, dimension(:) :: index_types
    integer, allocatable, dimension(:,:) :: type_indices
    integer, allocatable, dimension(:) :: ntype_indices
    logical, allocatable, dimension(:, :) :: mask
    integer, allocatable, dimension(:) :: tmp_nuclear_charges
    double precision, allocatable, dimension(:,:) :: tmp_coordinates

    integer, allocatable, dimension(:,:) :: start_indices
    double precision, allocatable, dimension(:) :: bag


    natypes = size(nasize, dim=1)
    natoms = size(nuclear_charges, dim=1)
    ! Allocate temporary
    allocate(start_indices(4,(natypes*(natypes+1)*(natypes+2)*(natypes+3))/24))
    ! Do consistency checks and get start_indices
    call global_checks(nuclear_charges, coordinates, charge_types, nsize, &
        & potential_types, sorting, nasize, natypes, nrep, start_indices)


    ! Allocate temporary
    allocate(distance_matrix(natoms,natoms))
    allocate(index_types(natoms))
    allocate(mask(natoms,natoms))
    allocate(decay_matrix(natoms,natoms))
    allocate(type_indices(natypes,natoms))
    allocate(ntype_indices(natypes))

    huge_double = huge(distance_matrix(1,1))

    ! Reorder nuclear_charges and coordinates if
    ! distance to center of mass sorting is used
    if (sorting == 2) then
        ! allocate temporary
        ! natoms size due to being used as temporary array later
        allocate(center_of_mass(natoms))
        allocate(tmp_coordinates(natoms,3))
        allocate(tmp_nuclear_charges(natoms))

        ! Get center of mass
        center_of_mass = 0.0d0
        do i=1, natoms
            center_of_mass(:3) = center_of_mass(:3) + nuclear_charges(i) * coordinates(i,:)
        enddo
        center_of_mass(:3) = center_of_mass(:3) / sum(nuclear_charges)

        do i=1, natoms
            tmp_coordinates(i,1) = sqrt(sum((coordinates(i,:) - center_of_mass(:3))**2))
        enddo

        ! Generate sorted list of atom ids by distance to center of mass
        do i = 1, natoms
            j = minloc(tmp_coordinates(:,1), dim=1)
            index_types(i) = j
            tmp_coordinates(j,1) = huge_double
        enddo

        tmp_coordinates = coordinates
        tmp_nuclear_charges = nuclear_charges
        do i=1, natoms
            j = index_types(i)
            coordinates(i,:) = tmp_coordinates(j,:)
            nuclear_charges(i) = tmp_nuclear_charges(j)
        enddo

        deallocate(center_of_mass)
        deallocate(tmp_coordinates)
        deallocate(tmp_nuclear_charges)
    endif

    ! Get distance matrix
    call get_distance_matrix(coordinates, distance_matrix)


    ! Get index types
    call get_index_types(charge_types, natypes, natoms, nuclear_charges, index_types)


    ! Create convenience arrays
    call get_convenience_arrays_global(distance_matrix, index_types, cutoff, decay_matrix, natypes, natoms, &
        & charge_types, nasize, ntype_indices, type_indices, decay, mask)

    if (potential_types(5) == 1) then
        ! Replace Rij with (Rij**3 + Vij**3)**(1/3)
        call smoothen_distance_matrix(natypes, natoms, charge_types, index_types, distance_matrix)
    endif

    ! allocate temporary
    allocate(one_body(natoms))

    one_body = 0.0d0
    ! One body interactions
    if (potential_types(1) > 0) then
        do i=1, natoms
            one_body(i) = weights_one(index_types(i)) * 0.5d0 * nuclear_charges(i)**2.4d0
        enddo
    endif

    ! allocate temporary
    allocate(two_body(natoms,natoms))

    two_body = 0.0d0

    if (sorting == 4) then
        ! get max bag size
        nbag = 0
        do i=1, natypes
            m = ntype_indices(i)
            nbag = max(nbag, m)
        enddo

        ! Allocate temporary
        allocate(bag(nbag))

        ! A bags
        do type1=1, natypes
            nbag = ntype_indices(type1)
            do n=1, ntype_indices(type1)
                i = type_indices(type1,n)
                bag(n) = one_body(i)
            enddo

            ! sort bag
            idx0 = start_indices(1,type1)
            do i = 1, nbag
                j = maxloc(bag(:nbag), dim=1)
                representation(idx0+i) = bag(j)
                bag(j) = -huge_double
            enddo
        enddo

        ! deallocate
        deallocate(bag)
    else
        do i=1, natoms
            two_body(i,i) = one_body(i)
        enddo
    endif

    ! deallocate
    deallocate(one_body)

    ! Two body interactions
    if (potential_types(2) > 0) then
        do i=1, natoms
            type1 = index_types(i)
            do j=i+1, natoms
                if (mask(i,j)) cycle
                type2 = index_types(j)

                norm = weights_two(type1, type2) * nuclear_charges(i) * nuclear_charges(j) &
                    & / distance_matrix(i,j) ** localization * decay_matrix(i,j)

                two_body(i,j) =  norm
                two_body(j,i) =  norm
            enddo
        enddo
    endif

    ! deallocate
    deallocate(index_types)

    if (sorting == 4) then
        ! bob sorting

        ! get max bag size
        nbag = 0
        do i=1, natypes
            m = ntype_indices(i)
            nbag = max(nbag, (m*(m-1))/2)
            do j=1, i
               l = ntype_indices(j)
               nbag = max(nbag, m*l)
            enddo
        enddo

        ! Allocate temporary
        allocate(bag(nbag))

        ! AA bags
        do type1=1, natypes
            nbag = (ntype_indices(type1)*ntype_indices(type1))/2
            do n=1, ntype_indices(type1)
                i = type_indices(type1,n)
                do m=1, n-1
                    j = type_indices(type1, m)
                    bag(n) = two_body(i,j)
                enddo
            enddo

            ! sort bag
            idx0 = start_indices(2,(type1*(type1+1))/2)
            do i = 1, nbag
                j = maxloc(bag(:nbag), dim=1)
                representation(idx0+i) = bag(j)
                bag(j) = -huge_double
            enddo
        enddo

        ! AB bags
        do type1=1, natypes
            do type2=1, type1 -1
                nbag = (ntype_indices(type1)*ntype_indices(type2))
                do n=1, ntype_indices(type1)
                    i = type_indices(type1,n)
                    do m=1, ntype_indices(type2)
                        j = type_indices(type2, m)
                        bag((n-1)*ntype_indices(type2) + m) = two_body(i,j)
                    enddo
                enddo

                ! sort bag
                idx0 = start_indices(2,(type1*(type1-1))/2 + type2)
                do i = 1, nbag
                    j = maxloc(bag(:nbag), dim=1)
                    representation(idx0+i) = bag(j)
                    bag(j) = -huge_double
                enddo
            enddo
        enddo

        ! deallocate
        deallocate(bag)
    endif

    if (potential_types(3) == 0 .AND. potential_types(4) == 0) then
        ! If there's no three or four bonding, finish up.
        if (sorting < 4) then
            ! Allocate temporary
            allocate(sorted_atoms(natoms))

            if (sorting == 3) then

                ! Allocate temporary
                allocate(row_norm(natoms))

                ! sort by row2 norm
                row_norm = sum(two_body**2, dim=1)
                do i = 1, natoms
                    j = maxloc(row_norm, dim=1)
                    sorted_atoms(i) = j
                    row_norm(j) = -huge_double
                enddo
                deallocate(row_norm)
            else
                do i=1, natoms
                    sorted_atoms(i) = i
                enddo
            endif

            representation = 0.0d0
            ! Fill one body
            if (potential_types(1) > 0) then
                do m = 1, natoms
                    i = sorted_atoms(m)
                    representation(m) = two_body(i,i)
                enddo
            endif
            ! Fill two body
            idx0 = start_indices(2,1)
            do m = 2, natoms
                i = sorted_atoms(m)
                idx = idx0 + ((m-1)*(m-2))/2
                do n = 1, m-1
                    j = sorted_atoms(n)
                    representation(idx+n) = two_body(i, j)
                enddo
            enddo

            deallocate(two_body)
            deallocate(sorted_atoms)
        endif
        return
    endif

    ! allocate temporary
    allocate(three_body(natoms,natoms,natoms))

    three_body = 0.0d0
    do i = 1, natoms
        three_body(i,i,i) = two_body(i,i)
        do j = 1, i-1
            norm = two_body(i,j)
            three_body(i,i,j) = norm
            three_body(i,j,i) = norm
            three_body(j,i,i) = norm
            three_body(j,j,i) = norm
            three_body(j,i,j) = norm
            three_body(i,j,j) = norm
        enddo
    enddo

    ! deallocate
    deallocate(two_body)

    if (potential_types(3) > 0) then
        ! divide by three_body weights to make sure two body terms are not reweighted
        three_body = three_body / weight_three
        call cm_global_three_body(three_body, potential_types(3), natoms, nuclear_charges, &
        & mask, decay_matrix, distance_matrix, localization, coordinates)
        ! weight three body terms
        three_body = three_body * weight_three
    endif

    if (sorting == 4) then
        ! bob sorting

        ! get max bag size
        nbag = 0
        do i=1, natypes
            m = ntype_indices(i)
            nbag = max(nbag, (m*(m-1)*(m-2))/2)
            do j=1, i
                l = ntype_indices(j)
                nbag = max(nbag, m*(m-1)*l)
                nbag = max(nbag, m*(l*(l-1))/2)
                do k=1, j-1
                    n = ntype_indices(k)
                    nbag = max(nbag, m*l*k)
                enddo
            enddo
        enddo

        ! Allocate temporary
        allocate(bag(nbag))

        if (potential_types(3) > 0 .AND. potential_types(3) < 5) then
            idx1 = 0
            do type1=1, natypes
                ! AAA bag
                idx1 = idx1+1
                idx = 0
                do n = 1, ntype_indices(type1)
                    i = type_indices(type1, n)
                    do m = 1, ntype_indices(type1)
                        if (n == m) cycle
                        j = type_indices(type1, m)
                        do l = 1, m-1
                            if (l == n) cycle
                            k = type_indices(type1, l)
                            idx = idx + 1
                            bag(idx) = three_body(i,j,k)
                        enddo
                    enddo
                enddo

                ! sort bag
                idx0 = start_indices(3,idx1)
                nbag = (ntype_indices(type1) * (ntype_indices(type1) - 1) * (ntype_indices(type1) - 2)) / 2
                do i = 1, nbag
                    j = maxloc(bag(:nbag), dim=1)
                    representation(idx0+i) = bag(j)
                    bag(j) = -huge_double
                enddo
                ! end AAA bag

                do type2 = 1, natypes
                    if (type1 == type2) cycle
                    ! AAB bag
                    idx1 = idx1+1
                    idx = 0
                    do n = 1, ntype_indices(type1)
                        i = type_indices(type1, n)
                        do m = 1, ntype_indices(type1)
                            if (n == m) cycle
                            j = type_indices(type1, m)
                            do l = 1, ntype_indices(type2)
                                k = type_indices(type2, l)
                                idx = idx + 1
                                bag(idx) = three_body(i,j,k)
                            enddo
                        enddo
                    enddo

                    ! sort bag
                    idx0 = start_indices(3,idx1)
                    nbag = ntype_indices(type1) * (ntype_indices(type1) - 1) * ntype_indices(type2)
                    do i = 1, nbag
                        j = maxloc(bag(:nbag), dim=1)
                        representation(idx0+i) = bag(j)
                        bag(j) = -huge_double
                    enddo
                    ! end AAB bag

                    ! ABB bag
                    idx1 = idx1+1
                    idx = 0
                    do n = 1, ntype_indices(type1)
                        i = type_indices(type1, n)
                        do m = 1, ntype_indices(type2)
                            j = type_indices(type2, m)
                            do l = 1, m-1
                                k = type_indices(type2, l)
                                idx = idx + 1
                                bag(idx) = three_body(i,j,k)
                            enddo
                        enddo
                    enddo

                    ! sort bag
                    idx0 = start_indices(3,idx1)
                    nbag = ntype_indices(type1) * (ntype_indices(type2) * (ntype_indices(type2)-1))/2
                    do i = 1, nbag
                        j = maxloc(bag(:nbag), dim=1)
                        representation(idx0+i) = bag(j)
                        bag(j) = -huge_double
                    enddo
                    ! end ABB bag

                    ! ABC bag
                    do type3=type2+1, natypes
                        if (type3 == type1) cycle
                        idx1 = idx1+1
                        idx = 0
                        do n = 1, ntype_indices(type1)
                            i = type_indices(type1, n)
                            do m = 1, ntype_indices(type2)
                                j = type_indices(type2, m)
                                do l = 1, ntype_indices(type3)
                                    k = type_indices(type3, l)
                                    idx = idx + 1
                                    bag(idx) = three_body(i,j,k)
                                enddo
                            enddo
                        enddo

                        ! sort bag
                        idx0 = start_indices(3,idx1)
                        nbag = ntype_indices(type1) * ntype_indices(type2) * ntype_indices(type3)
                        do i = 1, nbag
                            j = maxloc(bag(:nbag), dim=1)
                            representation(idx0+i) = bag(j)
                            bag(j) = -huge_double
                        enddo
                    enddo
                    ! end ABC bag
                enddo
            enddo
        else
            idx1 = 0
            do type1=1, natypes
                ! AAA bag
                idx1 = idx1+1
                idx = 0
                do n = 3, ntype_indices(type1)
                    i = type_indices(type1, n)
                    do m = 2, n-1
                        j = type_indices(type1, m)
                        do l = 1, m-1
                            k = type_indices(type1, l)
                            idx = idx + 1
                            bag(idx) = three_body(i,j,k)
                        enddo
                    enddo
                enddo

                ! sort bag
                idx0 = start_indices(3,idx1)
                nbag = (ntype_indices(type1) * (ntype_indices(type1) - 1) * (ntype_indices(type1) - 2)) / 6
                do i = 1, nbag
                    j = maxloc(bag(:nbag), dim=1)
                    representation(idx0+i) = bag(j)
                    bag(j) = -huge_double
                enddo
                ! end AAA bag

                do type2=type1+1, natypes
                    ! AAB bag
                    idx1 = idx1+1
                    idx = 0
                    do n = 2, ntype_indices(type1)
                        i = type_indices(type1, n)
                        do m = 1, n-1
                            j = type_indices(type1, m)
                            do l = 1, ntype_indices(type2)
                                k = type_indices(type2, l)
                                idx = idx + 1
                                bag(idx) = three_body(i,j,k)
                            enddo
                        enddo
                    enddo

                    ! sort bag
                    idx0 = start_indices(3,idx1)
                    nbag = (ntype_indices(type1) * (ntype_indices(type1) - 1))/2 * ntype_indices(type2)
                    do i = 1, nbag
                        j = maxloc(bag(:nbag), dim=1)
                        representation(idx0+i) = bag(j)
                        bag(j) = -huge_double
                    enddo
                    ! end AAB bag

                    ! ABB bag
                    idx1 = idx1+1
                    idx = 0
                    do n = 1, ntype_indices(type1)
                        i = type_indices(type1, n)
                        do m = 1, ntype_indices(type2)
                            j = type_indices(type2, m)
                            do l = 1, m-1
                                k = type_indices(type2, l)
                                idx = idx + 1
                                bag(idx) = three_body(i,j,k)
                            enddo
                        enddo
                    enddo

                    ! sort bag
                    idx0 = start_indices(3,idx1)
                    nbag = ntype_indices(type1) * (ntype_indices(type2) * (ntype_indices(type2)-1))/2
                    do i = 1, nbag
                        j = maxloc(bag(:nbag), dim=1)
                        representation(idx0+i) = bag(j)
                        bag(j) = -huge_double
                    enddo
                    ! end ABB bag

                    ! ABC bag
                    do type3=type2+1, natypes
                        idx1 = idx1+1
                        idx = 0
                        do n = 1, ntype_indices(type1)
                            i = type_indices(type1, n)
                            do m = 1, ntype_indices(type2)
                                j = type_indices(type2, m)
                                do l = 1, ntype_indices(type3)
                                    k = type_indices(type3, l)
                                    idx = idx + 1
                                    bag(idx) = three_body(i,j,k)
                                enddo
                            enddo
                        enddo

                        ! sort bag
                        idx0 = start_indices(3,idx1)
                        nbag = ntype_indices(type1) * ntype_indices(type2) * ntype_indices(type3)
                        do i = 1, nbag
                            j = maxloc(bag(:nbag), dim=1)
                            representation(idx0+i) = bag(j)
                            bag(j) = -huge_double
                        enddo
                    enddo
                    ! end ABC bag
                enddo
            enddo
        endif
    endif
    if (potential_types(4) == 0) then
        ! No four body, so finish up
        if (sorting < 4) then
            allocate(sorted_atoms(natoms))
            if (sorting == 3) then
                allocate(row_norm(natoms))
                ! sort by row2 norm
                row_norm = sum(sum(three_body**2, dim=1),dim=1)
                do i = 1, natoms
                    j = maxloc(row_norm, dim=1)
                    sorted_atoms(i) = j
                    row_norm(j) = -huge_double
                enddo
                deallocate(row_norm)
            else
                do i=1, natoms
                    sorted_atoms(i) = i
                enddo
            endif

            representation = 0.0d0
            ! Fill one body
            if (potential_types(1) > 0) then
                do m = 1, natoms
                    i = sorted_atoms(m)
                    representation(m) = three_body(i,i,i)
                enddo
            endif

            ! Fill two body
            idx0 = start_indices(2,1)
            do m = 2, natoms
                i = sorted_atoms(m)
                idx = idx0 + ((m-1)*(m-2))/2
                do n = 1, m-1
                    j = sorted_atoms(n)
                    representation(idx+n) = three_body(i, i, j)
                enddo
            enddo


            ! The representation size depends on the specific potential
            if (potential_types(3) < 5) then
                ! Fill three body
                idx0 = start_indices(3,1)
                do m = 1, natoms
                    i = sorted_atoms(m)
                    do n = 2, m-1
                        j = sorted_atoms(n)
                        do l=1, n-1
                            k = sorted_atoms(l)
                            idx = idx + 1
                            representation(idx) = three_body(i, j, k)
                        enddo
                    enddo
                    do n = m+1, natoms
                        j = sorted_atoms(n)
                        do l=1, m-1
                            k = sorted_atoms(l)
                            idx = idx + 1
                            representation(idx) = three_body(i, j, k)
                        enddo
                        do l=m+1, n-1
                            k = sorted_atoms(l)
                            idx = idx + 1
                            representation(idx) = three_body(i, j, k)
                        enddo
                    enddo
                enddo
            else
                ! Fill three body
                idx0 = start_indices(3,1)
                do m = 2, natoms
                    i = sorted_atoms(m)
                    do n = 1, m-1
                        j = sorted_atoms(n)
                        do l=1, n-1
                            k = sorted_atoms(l)
                            idx = idx + 1
                            representation(idx) = three_body(i, j, k)
                        enddo
                    enddo
                enddo
            endif
            deallocate(sorted_atoms)
        endif
        deallocate(three_body)
        return
    endif

    ! allocate temporary
    allocate(four_body(natoms,natoms,natoms,natoms))

    four_body = 0.0d0
    !$OMP PARALLEL DO PRIVATE(norm) SCHEDULE(dynamic)
    do i = 1, natoms
        four_body(i,i,i,i) = three_body(i,i,i)
        do j = 1, i-1
            norm = three_body(i,i,j)
            four_body(i,i,i,j) = norm
            four_body(i,i,j,i) = norm
            four_body(i,j,i,i) = norm
            four_body(j,i,i,i) = norm
            four_body(j,j,i,i) = norm
            four_body(j,i,j,i) = norm
            four_body(j,i,i,j) = norm
            four_body(i,j,j,i) = norm
            four_body(i,j,i,j) = norm
            four_body(i,i,j,j) = norm
            four_body(j,j,j,i) = norm
            four_body(j,j,i,j) = norm
            four_body(j,i,j,j) = norm
            four_body(i,j,j,j) = norm
            do k = 1, j-1
                norm = three_body(i,j,k)
                four_body(i,i,j,k) = norm
                four_body(i,i,k,j) = norm
                four_body(i,j,i,k) = norm
                four_body(i,j,j,k) = norm
                four_body(i,j,k,i) = norm
                four_body(i,j,k,j) = norm
                four_body(i,j,k,k) = norm
                four_body(i,k,i,j) = norm
                four_body(i,k,j,i) = norm
                four_body(i,k,j,j) = norm
                four_body(i,k,j,k) = norm
                four_body(i,k,k,j) = norm

                norm = three_body(j,i,k)
                four_body(j,i,i,k) = norm
                four_body(j,i,j,k) = norm
                four_body(j,i,k,i) = norm
                four_body(j,i,k,j) = norm
                four_body(j,i,k,k) = norm
                four_body(j,j,i,k) = norm
                four_body(j,j,k,i) = norm
                four_body(j,k,i,i) = norm
                four_body(j,k,i,j) = norm
                four_body(j,k,i,k) = norm
                four_body(j,k,j,i) = norm
                four_body(j,k,k,i) = norm

                norm = three_body(k,i,j)
                four_body(k,i,i,j) = norm
                four_body(k,i,j,i) = norm
                four_body(k,i,j,j) = norm
                four_body(k,i,j,k) = norm
                four_body(k,i,k,j) = norm
                four_body(k,j,i,i) = norm
                four_body(k,j,i,j) = norm
                four_body(k,j,i,k) = norm
                four_body(k,j,j,i) = norm
                four_body(k,j,k,i) = norm
                four_body(k,k,i,j) = norm
                four_body(k,k,j,i) = norm
            enddo
        enddo
    enddo

    ! deallocate
    deallocate(three_body)


    four_body = four_body / weight_four
    call cm_global_four_body(four_body, potential_types(4), natoms, nuclear_charges, &
    & mask, decay_matrix, distance_matrix)
    four_body = four_body * weight_four

    ! deallocate
    deallocate(distance_matrix)
    deallocate(mask)
    deallocate(decay_matrix)

    ! finish up
    if (sorting < 4) then
        allocate(sorted_atoms(natoms))
        if (sorting == 3) then
            allocate(row_norm(natoms))
            ! sort by row2 norm
            row_norm = sum(sum(sum(four_body**2, dim=1),dim=1),dim=1)
            do i = 1, natoms
                j = maxloc(row_norm, dim=1)
                sorted_atoms(i) = j
                row_norm(j) = -huge_double
            enddo
            deallocate(row_norm)
        else
            do i=1, natoms
                sorted_atoms(i) = i
            enddo
        endif

        representation = 0.0d0
        ! Fill one body
        if (potential_types(1) > 0) then
            do m = 1, natoms
                i = sorted_atoms(m)
                representation(m) = four_body(i,i,i,i)
            enddo
        endif

        ! Fill two body
        idx0 = start_indices(2,1)
        do m = 2, natoms
            i = sorted_atoms(m)
            idx = idx0 + ((m-1)*(m-2))/2
            do n = 1, m-1
                j = sorted_atoms(n)
                representation(idx+n) = four_body(i, i, i, j)
            enddo
        enddo


        ! The representation size depends on the specific potential
        if (potential_types(3) < 5) then
            ! Fill three body
            idx = start_indices(3,1)
            do m = 1, natoms
                i = sorted_atoms(m)
                do n = 2, m-1
                    j = sorted_atoms(n)
                    do l=1, n-1
                        k = sorted_atoms(l)
                        idx = idx + 1
                        representation(idx) = four_body(i, i, j, k)
                    enddo
                enddo
                do n = m+1, natoms
                    j = sorted_atoms(n)
                    do l=1, m-1
                        k = sorted_atoms(l)
                        idx = idx + 1
                        representation(idx) = four_body(i, i, j, k)
                    enddo
                    do l=m+1, n-1
                        k = sorted_atoms(l)
                        idx = idx + 1
                        representation(idx) = four_body(i, i, j, k)
                    enddo
                enddo
            enddo
        else
            ! Fill three body
            idx = start_indices(3,1)
            do m = 2, natoms
                i = sorted_atoms(m)
                do n = 1, m-1
                    j = sorted_atoms(n)
                    do l=1, n-1
                        k = sorted_atoms(l)
                        idx = idx + 1
                        representation(idx) = four_body(i, i, j, k)
                    enddo
                enddo
            enddo
        endif

        ! The representation size depends on the specific potential
        if (potential_types(4) < 3) then
            ! Fill four body
            idx = start_indices(4,1)
            do m = 1, natoms
                i = sorted_atoms(m)
                do n = 1, natoms
                    if (n == m) cycle
                    j = sorted_atoms(n)
                    do l=2, natoms
                        if (l == m .OR. l == n) cycle
                        k = sorted_atoms(l)
                        do a=1, l-1
                            if (a == m .OR. a == n) cycle
                            b = sorted_atoms(a)
                            idx = idx + 1
                            representation(idx) = four_body(i, j, k, b)
                        enddo
                    enddo
                enddo
            enddo
        else if (potential_types(4) < 5) then
            ! Fill four body
            idx = start_indices(4,1)
            do m = 1, natoms
                i = sorted_atoms(m)
                do n = 3, natoms
                    if (n == m) cycle
                    j = sorted_atoms(n)
                    do l=2, n-1
                        if (l == m) cycle
                        k = sorted_atoms(l)
                        do a=1, l-1 
                            if (a == m) cycle
                            b = sorted_atoms(a)
                            idx = idx + 1
                            representation(idx) = four_body(i, j, k, b)
                        enddo
                    enddo
                enddo
            enddo
        else if (potential_types(4) < 7) then
            ! Fill four body
            idx = start_indices(4,1)
            do m = 3, natoms
                i = sorted_atoms(m)
                do n = 1, natoms
                    if (n == m) cycle
                    j = sorted_atoms(n)
                    do l=2, m-1
                        if (l == m) cycle
                        k = sorted_atoms(l)
                        do a=1, l-1 
                            if (a == m) cycle
                            b = sorted_atoms(a)
                            idx = idx + 1
                            representation(idx) = four_body(i, j, k, b)
                        enddo
                    enddo
                enddo
            enddo
        else
            ! Fill four body
            idx = start_indices(4,1)
            do m = 4, natoms
                i = sorted_atoms(m)
                do n = 3, m-1
                    j = sorted_atoms(n)
                    do l=2, n-1
                        k = sorted_atoms(l)
                        do a=1, l-1 
                            b = sorted_atoms(a)
                            idx = idx + 1
                            representation(idx) = four_body(i, j, k, b)
                        enddo
                    enddo
                enddo
            enddo
        endif

        deallocate(sorted_atoms)
    else
        ! bob sorting

        ! get max bag size
        nbag = 0
        do i=1, natypes
            m = ntype_indices(i)
            nbag = max(nbag, (m*(m-1)*(m-2)*(m-3))/2)
            do j=1, i-1
                l = ntype_indices(j)
                nbag = max(nbag, m*(m-1)*(m-2)*l)
                nbag = max(nbag, m*l*(m-1)*(l-1))
                nbag = max(nbag, l*(l-1)*(l-2)*m)
                do k=1, j-1
                    n = ntype_indices(k)
                    nbag = max(nbag, m*(m-1)*l*n)
                    nbag = max(nbag, n*(n-1)*l*m)
                    nbag = max(nbag, l*(l-1)*n*m)
                    do a=1, k-1
                        b = ntype_indices(a)
                        nbag = max(nbag, m*l*n*b)
                    enddo
                enddo
            enddo
        enddo

        ! Allocate temporary
        allocate(bag(nbag))

        if (potential_types(4) > 0 .AND. potential_types(4) < 3) then

            idx1 = 0
            do type1=1,natypes
                !AAAA bag
                idx1 = idx1 + 1
                idx = 0
                do n = 1, ntype_indices(type1)
                    i = type_indices(type1, n)
                    do m=1, ntype_indices(type1)
                        if (n == m) cycle
                        j = type_indices(type1, m)
                        do l = 2, ntype_indices(type1)
                            if (l == m .OR. l == n) cycle
                            k = type_indices(type1, l)
                            do a=1, l-1
                                if (a == m .OR. a == n) cycle
                                b = type_indices(type1, a)
                                idx = idx + 1
                                bag(idx) = four_body(i,j,k,b)
                            enddo
                        enddo
                    enddo
                enddo

                ! sort bag
                idx0 = start_indices(4,idx1)
                nbag = (ntype_indices(type1) * (ntype_indices(type1) - 1) * (ntype_indices(type1) - 2) * &
                    & (ntype_indices(type1) - 3)) / 2
                do i = 1, nbag
                    j = maxloc(bag(:nbag), dim=1)
                    representation(idx0+i) = bag(j)
                    bag(j) = -huge_double
                enddo
                ! end AAAA bag

                do type2=1, natypes
                    if (type1 == type2) cycle
                    ! AAAB bag
                    idx1 = idx1 + 1
                    idx = 0
                    do n = 1, ntype_indices(type1)
                        i = type_indices(type1, n)
                        do m=1, ntype_indices(type1)
                            if (n == m) cycle
                            j = type_indices(type1, m)
                            do l = 1, ntype_indices(type1)
                                if (l == m .OR. l == n) cycle
                                k = type_indices(type1, l)
                                do a=1, ntype_indices(type2)
                                    b = type_indices(type2,a)
                                    idx = idx + 1
                                    bag(idx) = four_body(i,j,k,b)
                                enddo
                            enddo
                        enddo
                    enddo

                    ! sort bag
                    idx0 = start_indices(4,idx1)
                    nbag = ntype_indices(type1) * (ntype_indices(type1) - 1) * (ntype_indices(type1) - 2) * ntype_indices(type2)
                    do i = 1, nbag
                        j = maxloc(bag(:nbag), dim=1)
                        representation(idx0+i) = bag(j)
                        bag(j) = -huge_double
                    enddo
                    ! end AAAB bag

                    ! AABB bag
                    idx1 = idx1 + 1
                    idx = 0
                    do n = 1, ntype_indices(type1)
                        i = type_indices(type1, n)
                        do m=1, ntype_indices(type1)
                            if (n == m) cycle
                            j = type_indices(type1, m)
                            do l = 2, ntype_indices(type2)
                                k = type_indices(type2, l)
                                do a=1, l-1
                                    b = type_indices(type2,a)
                                    idx = idx + 1
                                    bag(idx) = four_body(i,j,k,b)
                                enddo
                            enddo
                        enddo
                    enddo

                    ! sort bag
                    idx0 = start_indices(4,idx1)
                    nbag = ntype_indices(type1) * (ntype_indices(type1) - 1) * (ntype_indices(type2)*(ntype_indices(type2)-1))/2
                    do i = 1, nbag
                        j = maxloc(bag(:nbag), dim=1)
                        representation(idx0+i) = bag(j)
                        bag(j) = -huge_double
                    enddo
                    ! end AABB bag

                    ! ABBB bag
                    idx1 = idx1 + 1
                    idx = 0
                    do n = 1, ntype_indices(type1)
                        i = type_indices(type1, n)
                        do m=1, ntype_indices(type2)
                            j = type_indices(type2, m)
                            do l = 2, ntype_indices(type2)
                                if (l == m) cycle
                                k = type_indices(type2, l)
                                do a=1, l-1
                                    if (a == m) cycle
                                    b = type_indices(type2,a)
                                    idx = idx + 1
                                    bag(idx) = four_body(i,j,k,b)
                                enddo
                            enddo
                        enddo
                    enddo

                    ! sort bag
                    idx0 = start_indices(4,idx1)
                    nbag = ntype_indices(type1) * (ntype_indices(type2) * (ntype_indices(type2) - 1) * &
                        & (ntype_indices(type2) - 2)) / 2
                    do i = 1, nbag
                        j = maxloc(bag(:nbag), dim=1)
                        representation(idx0+i) = bag(j)
                        bag(j) = -huge_double
                    enddo
                    ! end ABBB bag

                    ! ABAA bag
                    idx1 = idx1 + 1
                    idx = 0
                    do n = 1, ntype_indices(type1)
                        i = type_indices(type1, n)
                        do m=1, ntype_indices(type2)
                            j = type_indices(type2, m)
                            do l = 2, ntype_indices(type1)
                                if (l == n) cycle
                                k = type_indices(type1, l)
                                do a=1, l-1
                                    if (a == l) cycle
                                    b = type_indices(type1,a)
                                    idx = idx + 1
                                    bag(idx) = four_body(i,j,k,b)
                                enddo
                            enddo
                        enddo
                    enddo

                    ! sort bag
                    idx0 = start_indices(4,idx1)
                    nbag = ntype_indices(type1) * ntype_indices(type2) * &
                        & ((ntype_indices(type1) - 1) * (ntype_indices(type1) - 2)) / 2
                    do i = 1, nbag
                        j = maxloc(bag(:nbag), dim=1)
                        representation(idx0+i) = bag(j)
                        bag(j) = -huge_double
                    enddo
                    ! end ABAA bag

                    ! ABAB bag
                    idx1 = idx1 + 1
                    idx = 0
                    do n = 1, ntype_indices(type1)
                        i = type_indices(type1, n)
                        do m=1, ntype_indices(type2)
                            j = type_indices(type2, m)
                            do l = 1, ntype_indices(type1)
                                if (l == n) cycle
                                k = type_indices(type1, l)
                                do a=1, ntype_indices(type2)
                                    if (a == m) cycle
                                    b = type_indices(type2,a)
                                    idx = idx + 1
                                    bag(idx) = four_body(i,j,k,b)
                                enddo
                            enddo
                        enddo
                    enddo

                    ! sort bag
                    idx0 = start_indices(4,idx1)
                    nbag = ntype_indices(type1) * ntype_indices(type2) * (ntype_indices(type1) - 1) * (ntype_indices(type2) - 1)
                    do i = 1, nbag
                        j = maxloc(bag(:nbag), dim=1)
                        representation(idx0+i) = bag(j)
                        bag(j) = -huge_double
                    enddo
                    ! end ABAB bag

                    do type3=1, natypes
                        if (type1 == type3 .OR. type2 == type3) cycle

                        ! AABC bag
                        idx1 = idx1 + 1
                        idx = 0
                        do n = 1, ntype_indices(type1)
                            i = type_indices(type1, n)
                            do m=1, ntype_indices(type1)
                                if (n == m) cycle
                                j = type_indices(type1, m)
                                do l = 1, ntype_indices(type2)
                                    k = type_indices(type2, l)
                                    do a=1, ntype_indices(type3)
                                        b = type_indices(type3,a)
                                        idx = idx + 1
                                        bag(idx) = four_body(i,j,k,b)
                                    enddo
                                enddo
                            enddo
                        enddo

                        ! sort bag
                        idx0 = start_indices(4,idx1)
                        nbag = ntype_indices(type1) * (ntype_indices(type1)-1) * ntype_indices(type2) * ntype_indices(type3)
                        do i = 1, nbag
                            j = maxloc(bag(:nbag), dim=1)
                            representation(idx0+i) = bag(j)
                            bag(j) = -huge_double
                        enddo
                        ! end AABC bag

                        ! ABAC bag
                        idx1 = idx1 + 1
                        idx = 0
                        do n = 1, ntype_indices(type1)
                            i = type_indices(type1, n)
                            do m=1, ntype_indices(type2)
                                j = type_indices(type2, m)
                                do l = 1, ntype_indices(type1)
                                    if (n == l) cycle
                                    k = type_indices(type1, l)
                                    do a=1, ntype_indices(type3)
                                        b = type_indices(type3,a)
                                        idx = idx + 1
                                        bag(idx) = four_body(i,j,k,b)
                                    enddo
                                enddo
                            enddo
                        enddo

                        ! sort bag
                        idx0 = start_indices(4,idx1)
                        nbag = ntype_indices(type1) * (ntype_indices(type1)-1) * ntype_indices(type2) * ntype_indices(type3)
                        do i = 1, nbag
                            j = maxloc(bag(:nbag), dim=1)
                            representation(idx0+i) = bag(j)
                            bag(j) = -huge_double
                        enddo
                        ! end ABAC bag

                        ! ABBC bag
                        idx1 = idx1 + 1
                        idx = 0
                        do n = 1, ntype_indices(type1)
                            i = type_indices(type1, n)
                            do m=1, ntype_indices(type2)
                                j = type_indices(type2, m)
                                do l = 1, ntype_indices(type2)
                                    if (m == l) cycle
                                    k = type_indices(type2, l)
                                    do a=1, ntype_indices(type3)
                                        b = type_indices(type3,a)
                                        idx = idx + 1
                                        bag(idx) = four_body(i,j,k,b)
                                    enddo
                                enddo
                            enddo
                        enddo

                        ! sort bag
                        idx0 = start_indices(4,idx1)
                        nbag = ntype_indices(type1) * ntype_indices(type2) * (ntype_indices(type2)-1) * ntype_indices(type3)
                        do i = 1, nbag
                            j = maxloc(bag(:nbag), dim=1)
                            representation(idx0+i) = bag(j)
                            bag(j) = -huge_double
                        enddo
                        ! end ABBC bag

                        ! ABCC bag
                        idx1 = idx1 + 1
                        idx = 0
                        do n = 1, ntype_indices(type1)
                            i = type_indices(type1, n)
                            do m=1, ntype_indices(type2)
                                j = type_indices(type2, m)
                                do l = 2, ntype_indices(type3)
                                    k = type_indices(type3, l)
                                    do a=1, l-1
                                        b = type_indices(type3,a)
                                        idx = idx + 1
                                        bag(idx) = four_body(i,j,k,b)
                                    enddo
                                enddo
                            enddo
                        enddo

                        ! sort bag
                        idx0 = start_indices(4,idx1)
                        nbag = ntype_indices(type1) * ntype_indices(type2) * (ntype_indices(type3) * &
                            & (ntype_indices(type3)-1)) / 2
                        do i = 1, nbag
                            j = maxloc(bag(:nbag), dim=1)
                            representation(idx0+i) = bag(j)
                            bag(j) = -huge_double
                        enddo
                        ! end ABCC bag

                        do type4=1, natypes
                            if (type1 == type4 .OR. type2 == type4 .OR. type3 == type4) cycle
                            ! ABCD bag
                            idx1 = idx1 + 1
                            idx = 0
                            do n = 1, ntype_indices(type1)
                                i = type_indices(type1, n)
                                do m=1, ntype_indices(type2)
                                    j = type_indices(type2, m)
                                    do l = 1, ntype_indices(type3)
                                        k = type_indices(type3, l)
                                        do a=1, ntype_indices(type4)
                                            b = type_indices(type4,a)
                                            idx = idx + 1
                                            bag(idx) = four_body(i,j,k,b)
                                        enddo
                                    enddo
                                enddo
                            enddo

                            ! sort bag
                            idx0 = start_indices(4,idx1)
                            nbag = ntype_indices(type1) * ntype_indices(type2) * ntype_indices(type3) * ntype_indices(type4)
                            do i = 1, nbag
                                j = maxloc(bag(:nbag), dim=1)
                                representation(idx0+i) = bag(j)
                                bag(j) = -huge_double
                            enddo
                            ! end ABCD bag
                        enddo
                    enddo
                enddo
            enddo
        else if (potential_types(4) < 5) then
            idx1 = 0
            do type1=1,natypes
                !AAAA bag
                idx1 = idx1 + 1
                idx = 0
                do n = 1, ntype_indices(type1)
                    i = type_indices(type1, n)
                    do m=3, ntype_indices(type1)
                        if (n == m) cycle
                        j = type_indices(type1, m)
                        do l = 2, m-1
                            if (l == n) cycle
                            k = type_indices(type1, l)
                            do a=1, l-1
                                if (a == n) cycle
                                b = type_indices(type1, a)
                                idx = idx + 1
                                bag(idx) = four_body(i,j,k,b)
                            enddo
                        enddo
                    enddo
                enddo

                ! sort bag
                idx0 = start_indices(4,idx1)
                nbag = (ntype_indices(type1) * (ntype_indices(type1) - 1) * (ntype_indices(type1) - 2) * &
                    & (ntype_indices(type1) - 3)) / 6
                do i = 1, nbag
                    j = maxloc(bag(:nbag), dim=1)
                    representation(idx0+i) = bag(j)
                    bag(j) = -huge_double
                enddo
                ! end AAAA bag

                do type2=1, natypes
                    if (type1 == type2) cycle
                    ! AAAB bag
                    idx1 = idx1 + 1
                    idx = 0
                    do n = 1, ntype_indices(type1)
                        i = type_indices(type1, n)
                        do m=2, ntype_indices(type1)
                            if (n == m) cycle
                            j = type_indices(type1, m)
                            do l = 1, m-1
                                if (l == n) cycle
                                k = type_indices(type1, l)
                                do a=1, ntype_indices(type2)
                                    b = type_indices(type2,a)
                                    idx = idx + 1
                                    bag(idx) = four_body(i,j,k,b)
                                enddo
                            enddo
                        enddo
                    enddo

                    ! sort bag
                    idx0 = start_indices(4,idx1)
                    nbag = (ntype_indices(type1) * (ntype_indices(type1) - 1) * (ntype_indices(type1) - 2))/2 &
                        & * ntype_indices(type2)
                    do i = 1, nbag
                        j = maxloc(bag(:nbag), dim=1)
                        representation(idx0+i) = bag(j)
                        bag(j) = -huge_double
                    enddo
                    ! end AAAB bag


                    ! AABB bag
                    idx1 = idx1 + 1
                    idx = 0
                    do n = 1, ntype_indices(type1)
                        i = type_indices(type1, n)
                        do m=1, ntype_indices(type1)
                            if (n == m) cycle
                            j = type_indices(type1, m)
                            do l = 2, ntype_indices(type2)
                                k = type_indices(type2, l)
                                do a=1, l-1
                                    b = type_indices(type2,a)
                                    idx = idx + 1
                                    bag(idx) = four_body(i,j,k,b)
                                enddo
                            enddo
                        enddo
                    enddo

                    ! sort bag
                    idx0 = start_indices(4,idx1)
                    nbag = ntype_indices(type1) * (ntype_indices(type1) - 1) * (ntype_indices(type2)*(ntype_indices(type2)-1))/2
                    do i = 1, nbag
                        j = maxloc(bag(:nbag), dim=1)
                        representation(idx0+i) = bag(j)
                        bag(j) = -huge_double
                    enddo
                    ! end AABB bag


                    ! ABBB bag
                    idx1 = idx1 + 1
                    idx = 0
                    do n = 1, ntype_indices(type1)
                        i = type_indices(type1, n)
                        do m=3, ntype_indices(type2)
                            j = type_indices(type2, m)
                            do l = 2, m-1
                                k = type_indices(type2, l)
                                do a=1, l-1
                                    b = type_indices(type2,a)
                                    idx = idx + 1
                                    bag(idx) = four_body(i,j,k,b)
                                enddo
                            enddo
                        enddo
                    enddo

                    ! sort bag
                    idx0 = start_indices(4,idx1)
                    nbag = ntype_indices(type1) * (ntype_indices(type2) * (ntype_indices(type2) - 1) * &
                        & (ntype_indices(type2) - 2)) / 6
                    do i = 1, nbag
                        j = maxloc(bag(:nbag), dim=1)
                        representation(idx0+i) = bag(j)
                        bag(j) = -huge_double
                    enddo
                    ! end ABBB bag


                    do type3=1, natypes
                        if (type1 == type3 .OR. type2 == type3) cycle

                        ! AABC bag
                        idx1 = idx1 + 1
                        idx = 0
                        do n = 1, ntype_indices(type1)
                            i = type_indices(type1, n)
                            do m=1, ntype_indices(type1)
                                if (n == m) cycle
                                j = type_indices(type1, m)
                                do l = 1, ntype_indices(type2)
                                    k = type_indices(type2, l)
                                    do a=1, ntype_indices(type3)
                                        b = type_indices(type3,a)
                                        idx = idx + 1
                                        bag(idx) = four_body(i,j,k,b)
                                    enddo
                                enddo
                            enddo
                        enddo

                        ! sort bag
                        idx0 = start_indices(4,idx1)
                        nbag = ntype_indices(type1) * (ntype_indices(type1)-1) * &
                            & ntype_indices(type2) * ntype_indices(type3)
                        do i = 1, nbag
                            j = maxloc(bag(:nbag), dim=1)
                            representation(idx0+i) = bag(j)
                            bag(j) = -huge_double
                        enddo
                        ! end AABC bag

                        ! ABBC bag
                        idx1 = idx1 + 1
                        idx = 0
                        do n = 1, ntype_indices(type1)
                            i = type_indices(type1, n)
                            do m=2, ntype_indices(type2)
                                j = type_indices(type2, m)
                                do l = 1, m-1
                                    k = type_indices(type2, l)
                                    do a=1, ntype_indices(type3)
                                        b = type_indices(type3,a)
                                        idx = idx + 1
                                        bag(idx) = four_body(i,j,k,b)
                                    enddo
                                enddo
                            enddo
                        enddo

                        ! sort bag
                        idx0 = start_indices(4,idx1)
                        nbag = ntype_indices(type1) * (ntype_indices(type2) * (ntype_indices(type2)-1))/2 * &
                            & ntype_indices(type3)
                        do i = 1, nbag
                            j = maxloc(bag(:nbag), dim=1)
                            representation(idx0+i) = bag(j)
                            bag(j) = -huge_double
                        enddo
                        ! end ABBC bag

                        do type4=1, natypes
                            if (type1 == type4 .OR. type2 == type4 .OR. type3 == type4) cycle
                            ! ABCD bag
                            idx1 = idx1 + 1
                            idx = 0
                            do n = 1, ntype_indices(type1)
                                i = type_indices(type1, n)
                                do m=1, ntype_indices(type2)
                                    j = type_indices(type2, m)
                                    do l = 1, ntype_indices(type3)
                                        k = type_indices(type3, l)
                                        do a=1, ntype_indices(type4)
                                            b = type_indices(type4,a)
                                            idx = idx + 1
                                            bag(idx) = four_body(i,j,k,b)
                                        enddo
                                    enddo
                                enddo
                            enddo

                            ! sort bag
                            idx0 = start_indices(4,idx1)
                            nbag = ntype_indices(type1) * ntype_indices(type2) * &
                                & ntype_indices(type3) * ntype_indices(type4)
                            do i = 1, nbag
                                j = maxloc(bag(:nbag), dim=1)
                                representation(idx0+i) = bag(j)
                                bag(j) = -huge_double
                            enddo
                            ! end ABCD bag
                        enddo
                    enddo
                enddo
            enddo
        else if (potential_types(4) < 7) then
            ! not sure if this will give the same as the else if block above
            ! Here i and j are just swapped in four_body
            idx1 = 0
            do type1=1,natypes
                !AAAA bag
                idx1 = idx1 + 1
                idx = 0
                do n = 1, ntype_indices(type1)
                    i = type_indices(type1, n)
                    do m=3, ntype_indices(type1)
                        if (n == m) cycle
                        j = type_indices(type1, m)
                        do l = 2, m-1
                            if (l == n) cycle
                            k = type_indices(type1, l)
                            do a=1, l-1
                                if (a == n) cycle
                                b = type_indices(type1, a)
                                idx = idx + 1
                                bag(idx) = four_body(j,i,k,b)
                            enddo
                        enddo
                    enddo
                enddo

                ! sort bag
                idx0 = start_indices(4,idx1)
                nbag = (ntype_indices(type1) * (ntype_indices(type1) - 1) * &
                    & (ntype_indices(type1) - 2) * (ntype_indices(type1) - 3)) / 6
                do i = 1, nbag
                    j = maxloc(bag(:nbag), dim=1)
                    representation(idx0+i) = bag(j)
                    bag(j) = -huge_double
                enddo
                ! end AAAA bag

                do type2=1, natypes
                    if (type1 == type2) cycle
                    ! AAAB bag
                    idx1 = idx1 + 1
                    idx = 0
                    do n = 1, ntype_indices(type1)
                        i = type_indices(type1, n)
                        do m=2, ntype_indices(type1)
                            if (n == m) cycle
                            j = type_indices(type1, m)
                            do l = 1, m-1
                                if (l == n) cycle
                                k = type_indices(type1, l)
                                do a=1, ntype_indices(type2)
                                    b = type_indices(type2,a)
                                    idx = idx + 1
                                    bag(idx) = four_body(j,i,k,b)
                                enddo
                            enddo
                        enddo
                    enddo

                    ! sort bag
                    idx0 = start_indices(4,idx1)
                    nbag = (ntype_indices(type1) * (ntype_indices(type1) - 1) * (ntype_indices(type1) - 2))/2 * &
                        & ntype_indices(type2)
                    do i = 1, nbag
                        j = maxloc(bag(:nbag), dim=1)
                        representation(idx0+i) = bag(j)
                        bag(j) = -huge_double
                    enddo
                    ! end AAAB bag


                    ! AABB bag
                    idx1 = idx1 + 1
                    idx = 0
                    do n = 1, ntype_indices(type1)
                        i = type_indices(type1, n)
                        do m=1, ntype_indices(type1)
                            if (n == m) cycle
                            j = type_indices(type1, m)
                            do l = 2, ntype_indices(type2)
                                k = type_indices(type2, l)
                                do a=1, l-1
                                    b = type_indices(type2,a)
                                    idx = idx + 1
                                    bag(idx) = four_body(j,i,k,b)
                                enddo
                            enddo
                        enddo
                    enddo

                    ! sort bag
                    idx0 = start_indices(4,idx1)
                    nbag = ntype_indices(type1) * (ntype_indices(type1) - 1) * &
                        & (ntype_indices(type2)*(ntype_indices(type2)-1))/2
                    do i = 1, nbag
                        j = maxloc(bag(:nbag), dim=1)
                        representation(idx0+i) = bag(j)
                        bag(j) = -huge_double
                    enddo
                    ! end AABB bag


                    ! ABBB bag
                    idx1 = idx1 + 1
                    idx = 0
                    do n = 1, ntype_indices(type1)
                        i = type_indices(type1, n)
                        do m=3, ntype_indices(type2)
                            j = type_indices(type2, m)
                            do l = 2, m-1
                                k = type_indices(type2, l)
                                do a=1, l-1
                                    b = type_indices(type2,a)
                                    idx = idx + 1
                                    bag(idx) = four_body(j,i,k,b)
                                enddo
                            enddo
                        enddo
                    enddo

                    ! sort bag
                    idx0 = start_indices(4,idx1)
                    nbag = ntype_indices(type1) * (ntype_indices(type2) * (ntype_indices(type2) - 1) * &
                        & (ntype_indices(type2) - 2)) / 6
                    do i = 1, nbag
                        j = maxloc(bag(:nbag), dim=1)
                        representation(idx0+i) = bag(j)
                        bag(j) = -huge_double
                    enddo
                    ! end ABBB bag


                    do type3=1, natypes
                        if (type1 == type3 .OR. type2 == type3) cycle

                        ! AABC bag
                        idx1 = idx1 + 1
                        idx = 0
                        do n = 1, ntype_indices(type1)
                            i = type_indices(type1, n)
                            do m=1, ntype_indices(type1)
                                if (n == m) cycle
                                j = type_indices(type1, m)
                                do l = 1, ntype_indices(type2)
                                    k = type_indices(type2, l)
                                    do a=1, ntype_indices(type3)
                                        b = type_indices(type3,a)
                                        idx = idx + 1
                                        bag(idx) = four_body(j,i,k,b)
                                    enddo
                                enddo
                            enddo
                        enddo

                        ! sort bag
                        idx0 = start_indices(4,idx1)
                        nbag = ntype_indices(type1) * (ntype_indices(type1)-1) * &
                            & ntype_indices(type2) * ntype_indices(type3)
                        do i = 1, nbag
                            j = maxloc(bag(:nbag), dim=1)
                            representation(idx0+i) = bag(j)
                            bag(j) = -huge_double
                        enddo
                        ! end AABC bag

                        ! ABBC bag
                        idx1 = idx1 + 1
                        idx = 0
                        do n = 1, ntype_indices(type1)
                            i = type_indices(type1, n)
                            do m=2, ntype_indices(type2)
                                j = type_indices(type2, m)
                                do l = 1, m-1
                                    k = type_indices(type2, l)
                                    do a=1, ntype_indices(type3)
                                        b = type_indices(type3,a)
                                        idx = idx + 1
                                        bag(idx) = four_body(j,i,k,b)
                                    enddo
                                enddo
                            enddo
                        enddo

                        ! sort bag
                        idx0 = start_indices(4,idx1)
                        nbag = ntype_indices(type1) * (ntype_indices(type2) * (ntype_indices(type2)-1))/2 * &
                            & ntype_indices(type3)
                        do i = 1, nbag
                            j = maxloc(bag(:nbag), dim=1)
                            representation(idx0+i) = bag(j)
                            bag(j) = -huge_double
                        enddo
                        ! end ABBC bag

                        do type4=1, natypes
                            if (type1 == type4 .OR. type2 == type4 .OR. type3 == type4) cycle
                            ! ABCD bag
                            idx1 = idx1 + 1
                            idx = 0
                            do n = 1, ntype_indices(type1)
                                i = type_indices(type1, n)
                                do m=1, ntype_indices(type2)
                                    j = type_indices(type2, m)
                                    do l = 1, ntype_indices(type3)
                                        k = type_indices(type3, l)
                                        do a=1, ntype_indices(type4)
                                            b = type_indices(type4,a)
                                            idx = idx + 1
                                            bag(idx) = four_body(j,i,k,b)
                                        enddo
                                    enddo
                                enddo
                            enddo

                            ! sort bag
                            idx0 = start_indices(4,idx1)
                            nbag = ntype_indices(type1) * ntype_indices(type2) * &
                                & ntype_indices(type3) * ntype_indices(type4)
                            do i = 1, nbag
                                j = maxloc(bag(:nbag), dim=1)
                                representation(idx0+i) = bag(j)
                                bag(j) = -huge_double
                            enddo
                            ! end ABCD bag
                        enddo
                    enddo
                enddo
            enddo
        else if (potential_types(4) > 6) then
            idx1 = 0
            do type1=1,natypes
                !AAAA bag
                idx1 = idx1 + 1
                idx = 0
                do n = 4, ntype_indices(type1)
                    i = type_indices(type1, n)
                    do m=3, n-1
                        j = type_indices(type1, m)
                        do l = 2, m-1
                            k = type_indices(type1, l)
                            do a=1, l-1
                                b = type_indices(type1, a)
                                idx = idx + 1
                                bag(idx) = four_body(i,j,k,b)
                            enddo
                        enddo
                    enddo
                enddo

                ! sort bag
                idx0 = start_indices(4,idx1)
                nbag = (ntype_indices(type1) * (ntype_indices(type1) - 1) * (ntype_indices(type1) - 2) * &
                    & (ntype_indices(type1) - 3)) / 24
                do i = 1, nbag
                    j = maxloc(bag(:nbag), dim=1)
                    representation(idx0+i) = bag(j)
                    bag(j) = -huge_double
                enddo
                ! end AAAA bag

                do type2=1, natypes
                    if (type1 == type2) cycle
                    ! AAAB bag
                    idx1 = idx1 + 1
                    idx = 0
                    do n = 3, ntype_indices(type1)
                        i = type_indices(type1, n)
                        do m=2, n-1
                            j = type_indices(type1, m)
                            do l = 1, m-1
                                k = type_indices(type1, l)
                                do a=1, ntype_indices(type2)
                                    b = type_indices(type2,a)
                                    idx = idx + 1
                                    bag(idx) = four_body(i,j,k,b)
                                enddo
                            enddo
                        enddo
                    enddo

                    ! sort bag
                    idx0 = start_indices(4,idx1)
                    nbag = (ntype_indices(type1) * (ntype_indices(type1) - 1) * &
                        & (ntype_indices(type1) - 2))/6 * ntype_indices(type2)
                    do i = 1, nbag
                        j = maxloc(bag(:nbag), dim=1)
                        representation(idx0+i) = bag(j)
                        bag(j) = -huge_double
                    enddo
                    ! end AAAB bag

                    ! AABB bag
                    idx1 = idx1 + 1
                    idx = 0
                    do n = 2, ntype_indices(type1)
                        i = type_indices(type1, n)
                        do m=1, n-1
                            j = type_indices(type1, m)
                            do l = 2, ntype_indices(type2)
                                k = type_indices(type2, l)
                                do a=1, l-1
                                    b = type_indices(type2,a)
                                    idx = idx + 1
                                    bag(idx) = four_body(i,j,k,b)
                                enddo
                            enddo
                        enddo
                    enddo

                    ! sort bag
                    idx0 = start_indices(4,idx1)
                    nbag = (ntype_indices(type1) * (ntype_indices(type1) - 1))/2 * &
                        & (ntype_indices(type2)*(ntype_indices(type2)-1))/2
                    do i = 1, nbag
                        j = maxloc(bag(:nbag), dim=1)
                        representation(idx0+i) = bag(j)
                        bag(j) = -huge_double
                    enddo
                    ! end AABB bag

                    do type3=1, natypes
                        if (type1 == type3 .OR. type2 == type3) cycle

                        ! AABC bag
                        idx1 = idx1 + 1
                        idx = 0
                        do n = 2, ntype_indices(type1)
                            i = type_indices(type1, n)
                            do m=1, n-1
                                j = type_indices(type1, m)
                                do l = 1, ntype_indices(type2)
                                    k = type_indices(type2, l)
                                    do a=1, ntype_indices(type3)
                                        b = type_indices(type3,a)
                                        idx = idx + 1
                                        bag(idx) = four_body(i,j,k,b)
                                    enddo
                                enddo
                            enddo
                        enddo

                        ! sort bag
                        idx0 = start_indices(4,idx1)
                        nbag = (ntype_indices(type1) * (ntype_indices(type1)-1))/2 * &
                            & ntype_indices(type2) * ntype_indices(type3)
                        do i = 1, nbag
                            j = maxloc(bag(:nbag), dim=1)
                            representation(idx0+i) = bag(j)
                            bag(j) = -huge_double
                        enddo
                        ! end AABC bag

                        do type4=1, natypes
                            if (type1 == type4 .OR. type2 == type4 .OR. type3 == type4) cycle
                            ! ABCD bag
                            idx1 = idx1 + 1
                            idx = 0
                            do n = 1, ntype_indices(type1)
                                i = type_indices(type1, n)
                                do m=1, ntype_indices(type2)
                                    j = type_indices(type2, m)
                                    do l = 1, ntype_indices(type3)
                                        k = type_indices(type3, l)
                                        do a=1, ntype_indices(type4)
                                            b = type_indices(type4,a)
                                            idx = idx + 1
                                            bag(idx) = four_body(i,j,k,b)
                                        enddo
                                    enddo
                                enddo
                            enddo

                            ! sort bag
                            idx0 = start_indices(4,idx1)
                            nbag = ntype_indices(type1) * ntype_indices(type2) * &
                                & ntype_indices(type3) * ntype_indices(type4)
                            do i = 1, nbag
                                j = maxloc(bag(:nbag), dim=1)
                                representation(idx0+i) = bag(j)
                                bag(j) = -huge_double
                            enddo
                            ! end ABCD bag
                        enddo
                    enddo
                enddo
            enddo
        endif
    endif

    deallocate(index_types)
    deallocate(four_body)
    deallocate(type_indices)
    deallocate(ntype_indices)

end subroutine fgenerate_global

