! MIT License
!
! Copyright (c) 2016 Anders Steen Christensen
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

subroutine fbinarize(A, gran)

    implicit none

    double precision, dimension(:,:), intent(in) :: A
    double precision, intent(in) :: gran
    double precision, allocatable, dimension(:,:), intent(out) :: B

    double precision, allocatable, dimension(:) :: max_values
    double precision :: prec

    integer, allocatable, dimension(:) :: nright
    integer :: na, i, nleft

    prec = 13.8155
    na = size(A, dim=1)
    nb = size(A, dim=2)

    nleft = ceiling(prec)

    ! Allocate temporary
    allocate(max_values(nb))
    allocate(nright(nb))

    max_values = maxval(A, dim=1)

    max_values = max_values / gran + prec

    nright = ceiling(max_values)

    ndim = sum(nright)

    ! Allocate output
    allocate(B(na, ndim))






end subroutine fbinarize

