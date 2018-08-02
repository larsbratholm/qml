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

function decay(r, rc) result(f)

    implicit none

    double precision, intent(in), dimension(:,:) :: r
    double precision, intent(in) :: rc
    double precision, allocatable, dimension(:, :) :: f

    integer :: natoms

    double precision, parameter :: pi = 4.0d0 * atan(1.0d0)

    natoms = size(r, dim=1)

    ! Allocate temporary
    allocate(f(natoms, natoms))

    ! Decaying function reaching 0 at rc
    f = 0.5d0 * (cos(pi * r / rc) + 1.0d0)

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
    b2 = d - b

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

subroutine two_body_other(distance_matrix, index_pairs, element_types, rdecay, &
        & Rs, eta, rcut, nelements, rep)

    implicit none

    double precision, intent(in), dimension(:, :) :: distance_matrix
    integer, intent(in), dimension(:, :) :: index_pairs
    integer, intent(in), dimension(:) :: element_types
    double precision, intent(in), dimension(:, :) :: rdecay
    double precision, intent(in), dimension(:) :: Rs
    double precision, intent(in) :: eta, rcut
    integer, intent(in) :: nelements
    double precision, intent(inout), dimension(:, :) :: rep

    integer :: n_index_pairs, i, j, idx0, k, natoms, n, nRs
    double precision :: r

    n_index_pairs = size(index_pairs, dim=1)
    natoms = size(distance_matrix, dim=1)
    nRs = size(Rs, dim=1)

    ! Having j before k makes it possible to collapse the loop
    ! and avoid reduction of rep
    !$OMP PARALLEL DO PRIVATE(idx0, n, r) COLLAPSE(2) SCHEDULE(dynamic)
    do i = 1, n_index_pairs
        do j = 1, 4
            idx0 = index_pairs(i, j)
            do k = 1, natoms
                if (ANY(index_pairs(i, :) == k)) cycle

                r = distance_matrix(idx0, k)

                if (r > rcut) cycle

                n = (element_types(k) - 1) * nRs + (j-1) * nelements * nRs + 1
                rep(i, n: n + nRs - 1) = rep(i, n: n + nRs - 1) + &
                    & exp(-eta * (r - Rs)**2) * rdecay(k, idx0)
            enddo
        enddo
    enddo
    !$OMP END PARALLEL DO

end subroutine two_body_other

subroutine three_body_coupling_coupling(coordinates, index_pairs, rep)

    implicit none

    double precision, intent(in), dimension(:, :) :: coordinates
    integer, intent(in), dimension(:, :) :: index_pairs
    double precision, intent(inout), dimension(:, :) :: rep

    integer :: n_index_pairs, i, j, k, l, idx0, idx1, idx2, n
    double precision :: angle

    n_index_pairs = size(index_pairs, dim=1)

    !$OMP PARALLEL DO PRIVATE(idx0, idx1, idx2, angle, n) COLLAPSE(2) SCHEDULE(dynamic)
    do i = 1, n_index_pairs
        do j = 1, 4
            idx0 = index_pairs(i, j)
            do k = j+1, 4
                idx1 = index_pairs(i, k)
                do l = k+1, 4
                    idx2 = index_pairs(i, l)
                    angle = calc_angle(coordinates(idx1, :), coordinates(idx0, :), coordinates(idx2, :))
                    n = j + k + l - 5
                    rep(i, n) = cos(angle)
                enddo
            enddo
        enddo
    enddo
    !$OMP END PARALLEL DO

end subroutine three_body_coupling_coupling

subroutine three_body_coupling_other(distance_matrix, coordinates, index_pairs, element_types, &
        & Rs, Ts, eta, zeta, rcut, nelements, rep)

    implicit none

    double precision, intent(in), dimension(:, :) :: distance_matrix
    double precision, intent(in), dimension(:, :) :: coordinates
    integer, intent(in), dimension(:, :) :: index_pairs
    integer, intent(in), dimension(:) :: element_types
    double precision, intent(in), dimension(:) :: Rs
    double precision, intent(in), dimension(:) :: Ts
    double precision, intent(in) :: eta, rcut, zeta
    integer, intent(in) :: nelements
    double precision, intent(inout), dimension(:, :) :: rep

    integer :: n_index_pairs, i, j, idx0, idx1, k, l, natoms, n, nRs
    integer :: nTs, m, p
    double precision :: r, angle
    double precision, allocatable, dimension(:) :: radial, angular

    double precision, parameter :: pi = 4.0d0 * atan(1.0d0)

    n_index_pairs = size(index_pairs, dim=1)
    natoms = size(distance_matrix, dim=1)
    nRs = size(Rs, dim=1)
    nTs = size(Ts, dim=1)

    allocate(radial(nRs))
    allocate(angular(nTs))

    ! Having j,k before l makes it possible to collapse the loop
    ! and avoid reduction of rep
    !$OMP PARALLEL DO PRIVATE(idx0, idx1, n, m, r, radial, angle, angular) COLLAPSE(2) SCHEDULE(dynamic)
    do i = 1, n_index_pairs
        do j = 1, 4
            idx0 = index_pairs(i, j)
            do k = j+1, 4
                idx1 = index_pairs(i, k)
                do l = 1, natoms
                    if (ANY(index_pairs(i, :) == l)) cycle

                    r = 0.5d0 * (distance_matrix(idx0, l) + distance_matrix(idx1, l))

                    if (r > rcut) cycle

                    angle = calc_angle(coordinates(idx1, :), coordinates(idx0, :), coordinates(l, :))

                    n = (element_types(l) - 1) * nRs * nTs + &
                        & ((j * (7 - j)) / 2 + k - 5) * nelements * nRs * nTs + 1
                    radial = exp(-eta * (r - Rs)**2) * 0.5d0 * (cos(pi * r / rcut) + 1.0d0)
                    angular = (2.0d0 * ((1.0d0 + cos(angle - Ts)) * 0.5d0)**zeta)
                    do p = 1, nTs
                        m = n + (p-1) * nRs
                        rep(i, m: m + nRs - 1) = rep(i, m: m + nRs - 1) + &
                            & radial * angular(p)
                    enddo
                enddo
            enddo
        enddo
    enddo
    !$OMP END PARALLEL DO

    deallocate(radial)
    deallocate(angular)

end subroutine three_body_coupling_other

subroutine three_body_other_other(distance_matrix, coordinates, index_pairs, element_types, &
        & rdecay, Rs, Ts, eta, zeta, rcut, nelements, rep)

    implicit none

    double precision, intent(in), dimension(:, :) :: distance_matrix
    double precision, intent(in), dimension(:, :) :: coordinates
    integer, intent(in), dimension(:, :) :: index_pairs
    integer, intent(in), dimension(:) :: element_types
    double precision, intent(in), dimension(:, :) :: rdecay
    double precision, intent(in), dimension(:) :: Rs
    double precision, intent(in), dimension(:) :: Ts
    double precision, intent(in) :: eta, rcut, zeta
    integer, intent(in) :: nelements
    double precision, intent(inout), dimension(:, :) :: rep

    integer :: n_index_pairs, i, j, idx0, k, l, natoms, n, nRs
    integer :: nTs, m, p, s, t
    double precision :: rjk, angle, rjl
    double precision, allocatable, dimension(:) :: radial, angular

    double precision, parameter :: pi = 4.0d0 * atan(1.0d0)

    n_index_pairs = size(index_pairs, dim=1)
    natoms = size(distance_matrix, dim=1)
    nRs = size(Rs, dim=1)
    nTs = size(Ts, dim=1)

    allocate(radial(nRs))
    allocate(angular(nTs))

    ! Having j,k before l makes it possible to collapse the loop
    ! and avoid reduction of rep
    !$OMP PARALLEL DO PRIVATE(idx0, s, t, n, m, rjk, rjl, radial, angle, angular) COLLAPSE(2) SCHEDULE(dynamic)
    do i = 1, n_index_pairs
        do j = 1, 4
            idx0 = index_pairs(i, j)
            do k = 1, natoms
                if (ANY(index_pairs(i, :) == k)) cycle
                rjk = distance_matrix(idx0, k)
                if (rjk > rcut) cycle

                do l = k+1, natoms
                    if (ANY(index_pairs(i, :) == l)) cycle

                    rjl = distance_matrix(idx0, l)

                    if (rjl > rcut) cycle

                    angle = calc_angle(coordinates(k, :), coordinates(idx0, :), coordinates(l, :))

                    s = min(element_types(k), element_types(l))
                    t = max(element_types(k), element_types(l))

                    n = (-(s * (s - 1))/2 + &
                        & s * nelements + &
                        & t - nelements - 1) * nRs * nTs + 1 + &
                        & (j-1) * (nelements * (nelements + 1)) / 2 * nRs * nTs
                    radial = exp(-eta * ((rjk+rjl)/2.0d0 - Rs)**2) * &
                        & rdecay(idx0, k) * rdecay(idx0, l)
                    angular = 2.0d0 * ((1.0d0 + cos(angle - Ts)) * 0.5d0)**zeta
                    do p = 1, nTs
                        m = n + (p-1) * nRs
                        rep(i, m: m + nRs - 1) = rep(i, m: m + nRs - 1) + &
                            & radial * angular(p)
                    enddo
                enddo
            enddo
        enddo
    enddo
    !$OMP END PARALLEL DO

    deallocate(radial)
    deallocate(angular)

end subroutine three_body_other_other

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

subroutine four_body(coordinates, index_pairs, rep)

    implicit none

    double precision, intent(in), dimension(:,:) :: coordinates
    integer, intent(in), dimension(:,:) :: index_pairs
    double precision, intent(inout), dimension(:,:) :: rep

    integer :: n_index_pairs, i

    n_index_pairs = size(index_pairs, dim=1)

    !$OMP PARALLEL DO
    do i = 1, n_index_pairs
        rep(i,1) = calc_cos_dihedral(coordinates(index_pairs(i, 1),:), &
                                   & coordinates(index_pairs(i, 2),:), &
                                   & coordinates(index_pairs(i, 3),:), &
                                   & coordinates(index_pairs(i, 4),:))
        write(*,*) rep(i,1)
    enddo
    !$OMP END PARALLEL DO

end subroutine four_body

end module jcoupling_utils

subroutine fgenerate_jcoupling(coordinates, nuclear_charges, elements, index_pairs, &
                          & Rs2, Rs3, Ts, eta2, eta3, zeta, rcut, acut, nindex_pairs, rep_size, rep)

    use jcoupling_utils, only: decay, get_distance_matrix, two_body_coupling, &
        get_element_types, two_body_other, three_body_coupling_coupling, &
        three_body_coupling_other, three_body_other_other, four_body

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

    integer :: nelements
    integer, allocatable, dimension(:) :: element_types
    double precision, allocatable, dimension(:, :) :: distance_matrix, rdecay

    double precision, parameter :: pi = 4.0d0 * atan(1.0d0)

    if (size(coordinates, dim=1) /= size(nuclear_charges, dim=1)) then
        write(*,*) "ERROR: Atom Centered Symmetry Functions creation"
        write(*,*) size(coordinates, dim=1), "coordinates, but", &
            & size(nuclear_charges, dim=1), "nuclear_charges!"
        stop
    endif

    nelements = size(elements, dim=1)

    ! Store element index of every atom
    element_types = get_element_types(nuclear_charges, elements)

    ! Get distance matrix
    distance_matrix = get_distance_matrix(coordinates)

    ! Get two body decay
    rdecay = decay(distance_matrix, rcut)

    ! Calculate two body terms between coupling atoms
    call two_body_coupling(distance_matrix, index_pairs, rep(:, :6))

    ! Calculate two body terms between a coupling atom and
    ! other atoms
    call two_body_other(distance_matrix, index_pairs, element_types, rdecay, Rs2, &
        & eta2, rcut, nelements, rep(:, 7:42))

    ! Get three body decay
    rdecay = decay(distance_matrix, acut)

    ! Calculate three body terms between three coupling atoms
    call three_body_coupling_coupling(coordinates, index_pairs, rep(:, 43:46))

    ! Calculate three body terms between two coupling atoms and
    ! other atoms
    call three_body_coupling_other(distance_matrix, coordinates, index_pairs, &
        & element_types, Rs3, Ts, eta3, zeta, acut, nelements, rep(:, 47:208))

    ! Calculate three body terms between a coupling atom and two
    ! other atoms
    call three_body_other_other(distance_matrix, coordinates, index_pairs, &
        & element_types, rdecay, Rs3, Ts, eta3, zeta, acut, nelements, rep(:, 209:424))

    ! Calculate the four body term between coupling atoms
    call four_body(coordinates, index_pairs, rep(:,425:425))

    write(*,*) rep(:, 425)

    deallocate(element_types)
    deallocate(distance_matrix)
    deallocate(rdecay)

end subroutine fgenerate_jcoupling

!TODO symmetric
