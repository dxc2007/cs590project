!>  \brief  Collection of routines used by various programs to sum movies
module sum_movie_routines

    implicit none

    contains


    !>  \brief  Sum frames of a movie, applying supplied shifts and optionally applying filters
    subroutine sum_movie(   frames,sum_image,pixel_size, &
                            first_frame_to_sum,last_frame_to_sum, &
                            apply_shifts,apply_drift_filter,apply_dose_filter, &
                            apply_ctf,ctf_already_applied,phase_flip_only, &
                            restore_power, &
                            ctf_wiener_filter, &
                            x_shifts,y_shifts, &
                            ctf, minimum_absolute_ctf_value, &
                            electron_dose,exposure_per_frame, pre_exposure_amount, &
                            frc_file,movie_number,score, &
                            ctf_wiener_filter_constant)
        use Images
        use ProgressBars
        use ElectronDoses
        use Curves
        use NumericTextFiles
        use ContrastTransferFunctions
        use StringManipulations, only : IntegerToString, RealToString
        use UsefulFunctions, only : IsOdd
        ! arguments
        type(Image),                                intent(inout)   ::  frames(:)               !<  Array of frame images to be summed. It is assumed that all frames of the movie are in this array.
        type(Image),                                intent(inout)   ::  sum_image               !<  Output sum image. Will be overwritten.
        real,                                       intent(in)      ::  pixel_size              !<  In Angstroms
        integer,                                    intent(in)      ::  first_frame_to_sum      !<  Start summing at this frame number (first frame in array is 1)
        integer,                                    intent(in)      ::  last_frame_to_sum       !<  Last frame to sum
        logical,                                    intent(in)      ::  apply_shifts
        logical,                                    intent(in)      ::  apply_drift_filter
        logical,                                    intent(in)      ::  apply_dose_filter
        logical,                                    intent(in)      ::  apply_ctf
        logical,                                    intent(in)      ::  ctf_already_applied
        logical,                                    intent(in)      ::  phase_flip_only
        logical,                                    intent(in)      ::  restore_power           !<  Restore power after filters to the power before filters were applied
        logical,                                    intent(in)      ::  ctf_wiener_filter       !<  Apply Wiener like filter (i.e. division by the squares of the CTFs) to the final sum
        real,                           optional,   intent(in)      ::  x_shifts(:)             !<  X shifts (pixels)
        real,                           optional,   intent(in)      ::  y_shifts(:)             !<  Y shifts (pixels)
        type(ContrastTransferFunction), optional,   intent(in)      ::  ctf(:)                  !<  Array of CTF objects. One per frame.
        real,                           optional,   intent(in)      ::  minimum_absolute_ctf_value  !<  Absolute CTF not allowed to go below this
        type(ElectronDose),             optional,   intent(in)      ::  electron_dose           !<  Electron dose object (must be already initialised)
        real,                           optional,   intent(in)      ::  exposure_per_frame      !<  Exposure per frame (e/A2)
        real,                           optional,   intent(in)      ::  pre_exposure_amount     !<  Amount of pre-exposure (e/A2)
        type(NumericTextFile),          optional,   intent(inout)   ::  frc_file                !<  Numeric text file to write FRC to. Must already be setup with header comments already written.
        integer,                        optional,   intent(in)      ::  movie_number            !<  Movie number to be written to the FRC file comments
        real,                           optional,   intent(out)     ::  score                   !<  Score of how well the alignment worked
        real,                           optional,   intent(in)      ::  ctf_wiener_filter_constant  !<  Constant to be added to the sum of squared CTF terms in the denominator of the Wiener-like filter
        ! private variables
        logical,    parameter   ::  report_progress = .true.
        integer                 ::  number_of_frames_to_sum
        type(ProgressBar)       ::  my_progress_bar
        integer                 ::  progress_counter
        integer                 ::  frame_counter
        type(Image)             ::  sum_image_2
        type(Curve)             ::  fsc_curve
        real,       allocatable ::  frc_x_data(:), frc_y_data(:)
        real                    ::  score_numerator, score_denominator
        integer                 ::  bin_counter
        real                    ::  temp_real(2)
        integer                 ::  i,j
        real                    ::  x,y
        real                    ::  x_sq,y_sq
        real                    ::  current_denominator
        real                    ::  current_critical_dose
        real                    ::  current_optimal_dose
        real                    ::  current_dose
        real                    ::  spa_freq, spa_freq_sq
        real                    ::  current_ctf
        real                    ::  azimuth
        real                    ::  x_drifts(size(x_shifts)), y_drifts(size(y_shifts))
        ! start work

        !
        ! Check arguments
        !
        if (apply_shifts) then
            if (present(x_shifts) .and. present(y_shifts)) then
                if (size(x_shifts) .ne. size(y_shifts) .or. size(x_shifts) .ne. size(frames)) then
                    call this_program%TerminateWithFatalError('sum_movie','Incorrect number of X and Y shifts')
                endif
            else
                call this_program%TerminateWithFatalError('sum_movie','X and Y shifts not supplied')
            endif
        endif
        if (apply_dose_filter) then
            if (.not. present(electron_dose)) then
                call this_program%TerminateWithFatalError('sum_movie','Electron dose not given')
            endif
            if (.not. present(exposure_per_frame)) then
                call this_program%TerminateWithFatalError('sum_movie','Exposure per frame not given')
            endif
            if (.not. present(pre_exposure_amount)) then
                call this_program%TerminateWithFatalError('sum_movie','Pre exposure amount not given')
            endif
        endif
        if (present(frc_file)) then
            if (.not. present(movie_number)) then
                call this_program%TerminateWithFatalError('sum_movie','Movie number not given')
            endif
        endif
        if (apply_ctf) then
            if (present(ctf)) then
                if (size(ctf) .ne. size(frames)) then
                    call this_program%TerminateWithFatalError('sum_movie','Wrong number of CTFs')
                endif
            else
                call this_program%TerminateWithFatalError('sum_movie','CTFs not supplied')
            endif
            if (.not. present(minimum_absolute_ctf_value)) then
                call this_program%TerminateWithFatalError('sum_movie','Minimum absolute CTF value not given')
            endif
        endif
        if (ctf_wiener_filter) then
            if (.not. present(ctf_wiener_filter_constant)) then
                call this_program%TerminateWithFatalError('sum_movie','CTF Wiener-like filter constant not given')
            endif
            if (restore_power) then
                call this_program%TerminateWithFatalError('sum_movie',&
                                    'Not sure what to do with both power restoration and CTF Wiener-like filter')
            endif
            if (.not. apply_ctf .and. .not. ctf_already_applied) then
                call this_program%TerminateWithFatalError('sum_movie','Why apply Wiener-like CTF filter when no CTF is applied?')
            endif
        endif

        !
        ! Power restoration for the drift filter not yet implemented
        !
        if (apply_drift_filter .and. restore_power) then
            call this_program%TerminateWithFatalError('sum_movie','Power restoration for the drift filter not yet implemented')
        endif

        !
        number_of_frames_to_sum = last_frame_to_sum - first_frame_to_sum + 1

        !
        ! Fourier transform frames if necessary
        !
        if (report_progress) then
            write(*,'(a)') 'Fourier transforming images...'
            call my_progress_bar%Begin(number_of_frames_to_sum)
            progress_counter = 0
        endif
        !$omp parallel default(shared) private(frame_counter)
        !$omp do
        do frame_counter =first_frame_to_sum,last_frame_to_sum
            if (frames(frame_counter)%IsInRealSpace()) then
                call frames(frame_counter)%ForwardFFT()
            endif
            if (apply_ctf) then
                call frames(frame_counter)%ApplyCTF(ctf(frame_counter),phase_flip_only,minimum_absolute_ctf_value)
            endif
            if (report_progress) then
                !$omp atomic
                progress_counter = progress_counter + 1
                call my_progress_bar%Update(progress_counter)
            endif
        enddo
        !$omp enddo
        !$omp end parallel
        if (report_progress) call my_progress_bar%Finish()

        ! Shift images & apply drift filter
        if (report_progress) then
            if (apply_shifts .and. (apply_drift_filter .or. apply_dose_filter)) then
                write(*,'(a)') 'Shifting images & applying filter(s)...'
            else if (apply_shifts) then
                write(*,'(a)') 'Shifting images...'
            endif
            call my_progress_bar%Begin(number_of_frames_to_sum)
            progress_counter = 0
        endif
        !$omp parallel default(shared) private(frame_counter)
        !$omp do
        do frame_counter=first_frame_to_sum,last_frame_to_sum
            !
            if (apply_shifts) then
                call frames(frame_counter)%PhaseShift(  x_shifts(frame_counter), &
                                                        y_shifts(frame_counter), &
                                                        0.0e0)
            endif
            !
            if (apply_drift_filter) then
                ! What is the estimated drift within this frame?
                if (frame_counter .eq. 1) then
                    x_drifts(frame_counter) = x_shifts(2) - x_shifts(1)
                    y_drifts(frame_counter) = y_shifts(2) - y_shifts(1)
                else if (frame_counter .eq. size(x_shifts)) then
                    x_drifts(frame_counter) = x_shifts(size(x_shifts)) &
                                            - x_shifts(size(x_shifts) - 1)
                    y_drifts(frame_counter) = y_shifts(size(x_shifts)) &
                                            - y_shifts(size(x_shifts) - 1)
                else
                    x_drifts(frame_counter) = 0.5e0 * (   x_shifts(frame_counter+1) &
                                                        - x_shifts(frame_counter-1))
                    y_drifts(frame_counter) = 0.5e0 * (   y_shifts(frame_counter+1) &
                                                        - y_shifts(frame_counter-1))
                endif
                ! apply the corresponding drift filter
                call frames(frame_counter)%ApplyDriftFilter(x_drifts(frame_counter), y_drifts(frame_counter), 0.0e0)
            endif
            !
            if (apply_dose_filter) then
                call electron_dose%ApplyDoseFilterToImage(  frames(frame_counter), &
                                                            dose_start=((frame_counter-1)*exposure_per_frame) + pre_exposure_amount, &
                                                            dose_finish=(frame_counter*exposure_per_frame) + pre_exposure_amount, &
                                                            pixel_size=pixel_size)
            endif
            if (report_progress) then
                !$omp atomic
                progress_counter = progress_counter + 1
                call my_progress_bar%Update(progress_counter)
            endif
        enddo
        !$omp enddo
        !$omp end parallel
        if (report_progress) call my_progress_bar%Finish()

        !
        ! Prepare to compute the sum of the movie
        !
        call sum_image%Allocate(mould=frames(1))
        sum_image = (0e0, 0e0)

        if (.not. present(score) .and. .not. present(frc_file)) then
            do frame_counter=first_frame_to_sum,last_frame_to_sum
                call sum_image%AddImage(frames(frame_counter))
            enddo
        else
            !
            ! We need to compute two sums and FRC them
            !
            call sum_image_2%Allocate(mould=sum_image)
            sum_image_2 = (0e0, 0e0)

            do frame_counter=first_frame_to_sum,last_frame_to_sum
                if (IsOdd(frame_counter)) then
                    call sum_image  %AddImage(frames(frame_counter))
                else
                    call sum_image_2%AddImage(frames(frame_counter))
                endif
            enddo

            if (number_of_frames_to_sum .gt. 1) then
                fsc_curve = sum_image%GetFSCWith(sum_image_2, 100, pixel_size)
                call sum_image%AddImage(sum_image_2)
                call fsc_curve%CopyXData(frc_x_data)
                call fsc_curve%CopyYData(frc_y_data)

                ! Work out the Final score as per original unblur
                score_numerator = 0.0e0
                score_denominator = 0.0e0
                do bin_counter = 1, 100
                    if (frc_x_data(bin_counter) .le. 0.05e0) then
                        score_numerator   = score_numerator + frc_y_data(bin_counter) * real(bin_counter - 1)
                        score_denominator = score_denominator + real(bin_counter - 1)
                    endif
                enddo
                if (present(score)) then
                    score = score_numerator / score_denominator
                endif
                if (present(frc_file)) then
                    ! write out the fsc_curve.
                    call frc_file%WriteCommentLine('Micrograph '//IntegerToString(movie_number)// &
                                                    '. Score: '//RealToString(score,3))
                    do bin_counter = 1, 100
                        temp_real(1) = frc_x_data(bin_counter)
                        temp_real(2) = frc_y_data(bin_counter)

                        call frc_file%WriteDataLine(temp_real)
                    enddo
                endif
            else
                if (present(score)) score = 1.0e0
            endif
        endif

        !
        ! If necessary, restore the power
        !
        if (restore_power .and. (       apply_dose_filter &
                                .or.    apply_drift_filter &
                                .or.    apply_ctf &
                                .or.    ctf_already_applied &
                                ) &
            .or. ctf_wiener_filter) then


            !$omp parallel default(none) &
            !$omp private(i,j,x,y,x_sq,y_sq) &
            !$omp private (frame_counter, current_denominator, current_critical_dose, current_dose, current_optimal_dose) &
            !$omp private (current_ctf,azimuth,spa_freq,spa_freq_sq) &
            !$omp shared (sum_image,apply_dose_filter,electron_dose,pixel_size,first_frame_to_sum,last_frame_to_sum) &
            !$omp shared (exposure_per_frame,pre_exposure_amount,apply_ctf,ctf_already_applied,ctf,minimum_absolute_ctf_value,phase_flip_only) &
            !$omp shared (ctf_wiener_filter,ctf_wiener_filter_constant,restore_power)
            !$omp do

            ! for this pixel work out what to divide by. first we need the radius

            do j=1, sum_image%GetLogicalDimension(2)
                y = (sum_image%LogicalIndexGivenPhysicalIndexInFourierSpace(j,2) * sum_image%fourier_voxel_size(2))
                y_sq = y**2
                do i=1, sum_image%physical_upper_bound_complex(1)
                    x = ((i-1) * sum_image%fourier_voxel_size(1))
                    x_sq = x**2

                    ! Spatial frequency for this pixel
                    spa_freq_sq = x_sq + y_sq
                    spa_freq = sqrt(spa_freq)
                    azimuth = atan2(y,x)

                    ! What's the denominator for this pixel?
                    current_denominator = 0.0e0

                    ! Dose filter
                    if (apply_dose_filter .and. restore_power) then
                        if (i .eq. 1 .and. j .eq. 1) then
                            current_critical_dose = critical_dose_at_dc
                        else
                            current_critical_dose = electron_dose%CriticalDose(spa_freq / pixel_size)
                        endif

                        current_optimal_dose = electron_dose%OptimalDoseGivenCriticalDose(current_critical_dose)
                    endif


                    !
                    ! Add up denominators for all frames
                    !
                    do frame_counter = first_frame_to_sum,last_frame_to_sum
                        if (apply_dose_filter .and. restore_power) then
                            current_dose = (real(frame_counter) * exposure_per_frame) + pre_exposure_amount
                            if (current_dose .lt. current_optimal_dose) then
                                current_denominator = current_denominator &
                                                    + electron_dose%DoseFilter(current_dose, current_critical_dose)**2
                            endif
                        endif
                        if (     ((apply_ctf .or. ctf_already_applied) .and. .not. phase_flip_only) &
                            .or. ctf_wiener_filter) then
                            current_ctf = ctf(frame_counter)%EvaluateAtSquaredSpatialFrequency(spa_freq_sq,azimuth,.false.)
                            if (abs(current_ctf) .lt. minimum_absolute_ctf_value .and. .not. ctf_wiener_filter) then
                                current_ctf = sign(minimum_absolute_ctf_value,current_ctf)
                            endif
                            current_denominator = current_denominator + current_ctf**2
                        endif
                    enddo

                    if (ctf_wiener_filter) then
                        sum_image%complex_values(i, j, 1) = sum_image%complex_values(i, j, 1) &
                                                           / (current_denominator+ctf_wiener_filter_constant)
                    else
                        ! Apply denominator - this should restore total power to pre-filter level
                        if (current_denominator .gt. 0.0) then
                            current_denominator = sqrt(current_denominator)
                            sum_image%complex_values(i, j, 1) = sum_image%complex_values(i, j, 1) / current_denominator
                        endif
                    endif

                enddo
              enddo

            !$omp enddo
            !$omp end parallel

       endif

    end subroutine sum_movie

end module sum_movie_routines
