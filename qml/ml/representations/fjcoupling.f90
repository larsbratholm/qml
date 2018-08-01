! MIT License
!
! Copyright (c) 2018 Lars Andersen Bratholm
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

module jcoupling_utils

    implicit none

contains

function decay(r, invrc, natoms) result(f)

    implicit none

    double precision, intent(in), dimension(:,:) :: r
    double precision, intent(in) :: invrc
    integer, intent(in) :: natoms
    double precision, allocatable, dimension(:, :) :: f

    double precision, parameter :: pi = 4.0d0 * atan(1.0d0)

    ! Allocate temporary
    allocate(f(natoms, natoms))

    ! Decaying function reaching 0 at rc
    f = 0.5d0 * (cos(pi * r * invrc) + 1.0d0)

end function decay

function calc_angle(a, b, c) result(angle)

    implicit none

    double precision, intent(in), dimension(3) :: a
    double precision, intent(in), dimension(3) :: b
    double precision, intent(in), dimension(3) :: c

    double precision, dimension(3) :: v1
    double precision, dimension(3) :: v2

    double precision :: cos_angle
    double precision :: angle

    v1 = a - b
    v2 = c - b

    v1 = v1 / norm2(v1)
    v2 = v2 / norm2(v2)

    cos_angle = dot_product(v1,v2)

    ! Clipping
    if (cos_angle > 1.0d0) cos_angle = 1.0d0
    if (cos_angle < -1.0d0) cos_angle = -1.0d0

    angle = acos(cos_angle)

end function calc_angle

function calc_cos_dihedral(a, b, c, d) result(cos_dihedral)

    implicit none

    double precision, intent(in), dimension(3) :: a
    double precision, intent(in), dimension(3) :: b
    double precision, intent(in), dimension(3) :: c
    double precision, intent(in), dimension(3) :: d

    double precision, dimension(3) :: b0
    double precision, dimension(3) :: b1
    double precision, dimension(3) :: b2
    double precision, dimension(3) :: v
    double precision, dimension(3) :: w
    double precision, dimension(3) :: cross

    double precision :: cos_dihedral, x, y

    b0 = a - b
    b1 = c - b
    b2 = d - c

    b1 = b1 / norm2(b1)

    v = b0 - dot_product(b0, b1) * b1
    w = b2 - dot_product(b2, b1) * b1

    x = dot_product(v, w)

    cross(1) = b1(2) * v(3) - b1(3) * v(2)
    cross(2) = b1(3) * v(1) - b1(1) * v(3)
    cross(3) = b1(1) * v(2) - b1(2) * v(1)

    y = dot_product(cross, w)

    cos_dihedral = x / sqrt(x**2 + y**2)


end function calc_cos_dihedral

function get_distance_matrix(coordinates) result(distance_matrix)

    implicit none

    double precision, intent(in), dimension(:,:) :: coordinates
    double precision, allocatable, dimension(:,:) :: distance_matrix

    integer :: i, j, natoms
    double precision :: rij

    natoms = size(coordinates, dim=1)

    ! Get distance matrix
    ! Allocate temporary
    allocate(distance_matrix(natoms, natoms))
    distance_matrix = 0.0d0

    !$OMP PARALLEL DO PRIVATE(rij) SCHEDULE(dynamic)
    do i = 1, natoms
        do j = i+1, natoms
            rij = norm2(coordinates(j,:) - coordinates(i,:))
            distance_matrix(i, j) = rij
            distance_matrix(j, i) = rij
        enddo
    enddo
    !$OMP END PARALLEL DO

end function get_distance_matrix

subroutine two_body_coupling(distance_matrix, index_pairs, rep)

    implicit none

    double precision, intent(in), dimension(:, :) :: distance_matrix
    integer, intent(in), dimension(:, :) :: index_pairs
    double precision, intent(inout), dimension(:, :) :: rep

    integer :: n_index_pairs, i, j, k, idx0, idx1, n

    n_index_pairs = size(index_pairs, dim=1)

    !$OMP PARALLEL DO PRIVATE(idx0, idx1, n) COLLAPSE(2) SCHEDULE(dynamic)
    do i = 1, n_index_pairs
        do j = 1, 4
            idx0 = index_pairs(i, j)
            do k = j+1, 4
                idx1 = index_pairs(i, k)
                n = (j * (7 - j)) / 2 + k - 4
                rep(i, n) = distance_matrix(idx0, idx1)
            enddo
        enddo
    enddo
    !$OMP END PARALLEL DO

end subroutine two_body_coupling

subroutine two_body_other(distance_matrix, index_pairs, element_types, Rs2, eta2, rcut, rep)

    implicit none

    double precision, intent(in), dimension(:, :) :: distance_matrix
    integer, intent(in), dimension(:, :) :: index_pairs
    integer, intent(in), dimension(:) :: element_types
    double precision, intent(in), dimension(:) :: Rs2
    double precision, intent(in) :: eta2, rcut
    double precision, intent(inout), dimension(:, :) :: rep

    integer :: n_index_pairs, i, j, idx0, k, natoms
    double precision :: r

    n_index_pairs = size(index_pairs, dim=1)
    natoms = size(distance_matrix, dim=1)

    ! Having j before k makes it possible to collapse the loop
    ! and avoid reduction of rep
    !$OMP PARALLEL DO PRIVATE(idx0, , n) COLLAPSE(2) SCHEDULE(dynamic)
    do i = 1, n_index_pairs
        do j = 1, 4
            idx0 = index_pairs(i, j)
            do k = 1, natoms
                if (ANY(index_pairs(i, :) == k)) cycle

                r = distance_matrix(idx0, k)
                n = 
                rep(i, n) = rep(i, n) + XXX
            enddo
        enddo
    enddo


def two_body_other(distances, idx, z, ele, basis, eta, rcut):
    def atom_to_other(idx0):
        arep = np.zeros((len(ele), nbasis))
        for idx1 in range(natoms):
            if idx1 in idx:
                continue
            d = distances[idx0, idx1]
            if d > rcut:
                continue
            ele_idx = np.where(z[idx1] == ele)[0][0]
            arep[ele_idx] += two_body_basis(d, basis, eta, rcut)
        return arep

    nbasis = len(basis)
    natoms = distances.shape[0]
    rep = []
    # two body term between coupling atoms and non-coupling
    for idx0 in idx:
        rep.append(atom_to_other(idx0))
    return rep


end subroutine two_body_other

function get_element_types(nuclear_charges, elements) result(element_types)

    implicit none

    integer, intent(in), dimension(:) :: nuclear_charges
    integer, intent(in), dimension(:) :: elements

    integer, allocatable, dimension(:) :: element_types

    integer :: natoms, nelements, i, j

    natoms = size(nuclear_charges, dim=1)
    nelements = size(elements, dim=1)

    ! Allocate temporary
    allocate(element_types(natoms))

    !$OMP PARALLEL DO
    do i = 1, natoms
        do j = 1, nelements
            if (nuclear_charges(i) .eq. elements(j)) then
                element_types(i) = j
                continue
            endif
        enddo
    enddo
    !$OMP END PARALLEL DO

end function get_element_types

end module jcoupling_utils

subroutine fgenerate_jcoupling(coordinates, nuclear_charges, elements, index_pairs, &
                          & Rs2, Rs3, Ts, eta2, eta3, zeta, rcut, acut, nindex_pairs, rep_size, rep)

    use jcoupling_utils, only: decay, calc_angle, get_distance_matrix, two_body_coupling, &
        get_element_types, calc_cos_dihedral

    implicit none

    double precision, intent(in), dimension(:, :) :: coordinates
    integer, intent(in), dimension(:) :: nuclear_charges
    integer, intent(in), dimension(:) :: elements
    integer, intent(in), dimension(:, :) :: index_pairs
    double precision, intent(in), dimension(:) :: Rs2
    double precision, intent(in), dimension(:) :: Rs3
    double precision, intent(in), dimension(:) :: Ts
    double precision, intent(in) :: eta2
    double precision, intent(in) :: eta3
    double precision, intent(in) :: zeta
    double precision, intent(in) :: rcut
    double precision, intent(in) :: acut
    integer, intent(in) :: nindex_pairs
    integer, intent(in) :: rep_size
    double precision, intent(out), dimension(nindex_pairs, rep_size) :: rep

    integer, allocatable, dimension(:) :: element_types
    double precision, allocatable, dimension(:, :) :: distance_matrix

    !integer :: k, l, n, m, p, q, s, z, nbasis3, nabasis, i, j
    !double precision :: rik, angle, rij
    !double precision, allocatable, dimension(:) :: radial, angular, a, b, c
    !double precision, allocatable, dimension(:, :) :: rdecay, rep3

    double precision, parameter :: pi = 4.0d0 * atan(1.0d0)

    if (size(coordinates, dim=1) /= size(nuclear_charges, dim=1)) then
        write(*,*) "ERROR: Atom Centered Symmetry Functions creation"
        write(*,*) size(coordinates, dim=1), "coordinates, but", &
            & size(nuclear_charges, dim=1), "nuclear_charges!"
        stop
    endif

    ! Store element index of every atom
    element_types = get_element_types(nuclear_charges, elements)

    ! Get distance matrix
    distance_matrix = get_distance_matrix(coordinates)

    ! Calculate two body terms between coupling atoms
    call two_body_coupling(distance_matrix, index_pairs, rep(:, :6))

    ! Calculate two body terms between a coupling atom and
    ! other atoms
    call two_body_other(distance_matrix, index_pairs, element_types, Rs2, eta2, rcut, rep(:, 7:42))

    !! number of basis functions in the two body term
    !nbasis2 = size(Rs2)

    !! Inverse of the two body cutoff
    !invcut = 1.0d0 / rcut
    !! pre-calculate the radial decay in the two body terms
    !rdecay = decay(distance_matrix, invcut, natoms)

    !! Allocate temporary
    !allocate(radial(nbasis2))

    !!$OMP PARALLEL DO PRIVATE(n,m,rij,radial) REDUCTION(+:rep)
    !do i = 1, natoms
    !    ! index of the element of atom i
    !    m = element_types(i)
    !    do j = i + 1, natoms
    !        ! index of the element of atom j
    !        n = element_types(j)
    !        ! distance between atoms i and j
    !        rij = distance_matrix(i,j)
    !        if (rij <= rcut) then
    !            ! two body term of the representation
    !            radial = exp(-eta2*(rij - Rs2)**2) * rdecay(i,j)
    !            rep(i, (n-1)*nbasis2 + 1:n*nbasis2) = rep(i, (n-1)*nbasis2 + 1:n*nbasis2) + radial
    !            rep(j, (m-1)*nbasis2 + 1:m*nbasis2) = rep(j, (m-1)*nbasis2 + 1:m*nbasis2) + radial
    !        endif
    !    enddo
    !enddo
    !!$OMP END PARALLEL DO

    !deallocate(radial)

    !! number of radial basis functions in the three body term
    !nbasis3 = size(Rs3)
    !! number of radial basis functions in the three body term
    !nabasis = size(Ts)

    !! Inverse of the three body cutoff
    !invcut = 1.0d0 / acut
    !! pre-calculate the radial decay in the three body terms
    !rdecay = decay(distance_matrix, invcut, natoms)

    !! Allocate temporary
    !allocate(rep3(natoms,rep_size))
    !allocate(a(3))
    !allocate(b(3))
    !allocate(c(3))
    !allocate(radial(nbasis3))
    !allocate(angular(nabasis))

    !rep3 = 0.0d0

    !! This could probably be done more efficiently if it's a bottleneck
    !! Also the order is a bit wobbly compared to the tensorflow implementation
    !!$OMP PARALLEL DO PRIVATE(rij, n, rik, m, a, b, c, angle, radial, angular, &
    !!$OMP p, q, s, z) REDUCTION(+:rep3) COLLAPSE(2) SCHEDULE(dynamic)
    !do i = 1, natoms
    !    do j = 1, natoms - 1
    !        if (i .eq. j) cycle
    !        ! distance between atoms i and j
    !        rij = distance_matrix(i,j)
    !        if (rij > acut) cycle
    !        ! index of the element of atom j
    !        n = element_types(j)
    !        do k = j + 1, natoms
    !            if (i .eq. k) cycle
    !            ! distance between atoms i and k
    !            rik = distance_matrix(i,k)
    !            if (rik > acut) cycle
    !            ! index of the element of atom k
    !            m = element_types(k)
    !            ! coordinates of atoms j, i, k
    !            a = coordinates(j,:)
    !            b = coordinates(i,:)
    !            c = coordinates(k,:)
    !            ! angle between atoms i, j and k centered on i
    !            angle = calc_angle(a,b,c)
    !            ! The radial part of the three body terms including decay
    !            radial = exp(-eta3*(0.5d0 * (rij+rik) - Rs3)**2) * rdecay(i,j) * rdecay(i,k)
    !            ! The angular part of the three body terms
    !            angular = 2.0d0 * ((1.0d0 + cos(angle - Ts)) * 0.5d0) ** zeta
    !            ! The lowest of the element indices for atoms j and k
    !            p = min(n,m) - 1
    !            ! The highest of the element indices for atoms j and k
    !            q = max(n,m) - 1
    !            ! calculate the indices that the three body terms should be added to
    !            s = nelements * nbasis2 + nbasis3 * nabasis * (nelements * p + q) + 1
    !            do l = 1, nbasis3
    !                ! calculate the indices that the three body terms should be added to
    !                z = s + (l-1) * nabasis
    !                ! Add the contributions from atoms i,j and k
    !                rep3(i, z:z + nabasis - 1) = rep3(i, z:z + nabasis - 1) + angular * radial(l)
    !            enddo
    !        enddo
    !    enddo
    !enddo
    !!$OMP END PARALLEL DO

    !rep = rep + rep3

    deallocate(element_types)
    deallocate(distance_matrix)
    !deallocate(rdecay)
    !deallocate(rep3)
    !deallocate(a)
    !deallocate(b)
    !deallocate(c)
    !deallocate(radial)
    !deallocate(angular)

end subroutine fgenerate_jcoupling
