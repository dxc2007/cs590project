!>  \brief  Look up tables for expensive functions such as sine, cosine etc
module LookUpTables
    implicit none
    private
    public :: sin_lookup, cos_lookup
    integer             ::  i
    real,   parameter   ::  pi                  =   acos(-1.0e0)
    real,   parameter   ::  two_pi              =   2.0e0*acos(-1.0e0)
    real,   parameter   ::  half_pi             =   0.5e0*acos(-1.0e0)
    real,   parameter   ::  pi_inverse          =   1.0e0 / acos(-1.0e0)
    real,   parameter   ::  sin_lut_50(0:50)    =   (/ (sin(real(i)*pi/100.0e0),i=0,50)   /)

    contains

    !>  \brief  Check look up tables are OK
    subroutine check_sin_lookup()
        ! private variables
        integer ::  i
        real    ::  angle
        !
        do i=-500,500
            angle = real(i)*pi/100.0
            if (abs(sin(angle) - sin_lookup(angle)) .gt. 0.01) then
                print *, i, angle, sin(angle), sin_lookup(angle)
                stop 'buggy lookup'
            endif
        enddo
        write(*,'(a)') 'check_sin_lookup: all good!'
    end subroutine check_sin_lookup

    !>  \brief  Look up the value of sine given an argument
    elemental real function sin_lookup(argument)
        real,   intent(in)  ::  argument
        ! private variables
        real                ::  angle
        integer             ::  lut_index
        ! start work
        angle = argument * pi_inverse ! angle as a fraction of pi
        angle = mod(angle,2.0e0) ! bring it back to between -2.0 and 2.0 !! THIS LINE IS EXPENSIVE
        if (angle .lt. 0.0e0) angle = angle + 2.0e0 ! angle is now between 0.0 and 2.0
        if (angle .le. 0.5e0) then
            ! sin(x)
            lut_index = nint(angle*100.0e0)
            sin_lookup = sin_lut_50(lut_index)
        else if (angle .le. 1.0e0) then
            ! sin(x) = sin(pi-x)
            angle = 1.0e0 - angle
            lut_index = nint(angle*100.0e0)
            sin_lookup = sin_lut_50(lut_index)
        else if (angle .le. 1.5e0) then
            ! sin(x) = -sin(x-pi)
            angle = angle - 1.0e0
            lut_index = nint(angle*100.0e0)
            sin_lookup = -sin_lut_50(lut_index)
        else
            ! sin(x) = -sin(2pi-x)
            angle = 2.0e0-angle
            lut_index = nint(angle*100.0e0)
            sin_lookup = -sin_lut_50(lut_index)
        endif
    end function sin_lookup

    !>  \brief  Look up the value of cosine
    elemental real function cos_lookup(argument)
        real,   intent(in)  ::  argument
        ! private variables
        ! cos(x) = sin(x+pi/2)
        cos_lookup = sin_lookup(argument+half_pi)
    end function cos_lookup

end module LookUpTables
