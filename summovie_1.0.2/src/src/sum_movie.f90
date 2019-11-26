program unblur

    use Globals


    use UserSuppliedParameters
    use UserInputs
    use ImageFiles
    use Images
    use Peaks
    use Curves
    use ProgressBars
    use NumericTextFiles
    use UsefulFunctions
    use StringManipulations
    use ElectronDoses
    use ContrastTransferFunctions
    use ctffind_routines

    use sum_movie_routines

    implicit none

    ! Variables associated with user input

    type(UserInput)            ::   my_user_input

    type(UserSuppliedInteger)  ::   user_supplied_first_frame
    type(UserSuppliedInteger)  ::   user_supplied_last_frame
    type(UserSuppliedReal)     ::   pixel_size
    type(UserSuppliedFilename) ::   user_supplied_input_filename
    type(UserSuppliedFilename) ::   user_supplied_output_filename
    type(UserSuppliedFilename) ::   user_supplied_shifts_input_filename
    type(UserSuppliedFilename) ::   user_supplied_frc_output_filename
    type(UserSuppliedLogical)  ::   user_supplied_apply_drift_filter
    type(UserSuppliedInteger)  ::   number_of_frames_per_movie
    type(UserSuppliedLogical)   ::  apply_dose_filter
    type(UserSuppliedReal)      ::  dose_per_frame
    type(UserSuppliedReal)          ::  pre_exposure_amount
    type(UserSuppliedReal)      ::  acceleration_voltage
    type(UserSuppliedLogical)   ::  restore_power
    type(UserSuppliedLogical)   ::  apply_ctf_wiener_filter
    type(UserSuppliedReal)      ::  ctf_wiener_filter_constant

    type(UserSuppliedLogical)       ::  apply_ctf
    type(UserSuppliedLogical)       ::  phase_flip_only
    type(UserSuppliedFilename)      ::  ctffind_results

    ! Variable associated with Images

    type(ImageFile)                 ::  input_imagefile

    type(Image), allocatable        ::  image_stack(:)
    type(Image)                     ::  sum_image
    type(Image)                     ::  sum_image_2

    real                            ::  current_drift_x
    real                            ::  current_drift_y

    integer                         ::  number_of_input_images
    integer                         ::  number_of_frames_to_sum
    integer                         ::  number_of_movies

    integer                         ::  image_counter
    integer                         ::  bin_counter
    integer                         ::  movie_counter
    integer                         ::  progress_counter


    type(ProgressBar)               ::  my_progress_bar

    type(NumericTextFile)           ::  my_shifts_file
    type(NumericTextFile)           ::  my_frc_file

    type(Curve)                     ::  fsc_curve


    real, allocatable               ::  total_x_shifts(:)
    real, allocatable               ::  total_y_shifts(:)

    real, allocatable               ::  frc_x_data(:)
    real, allocatable               ::  frc_y_data(:)

    real                            ::  temp_real(2) ! for writing shifts to file

    real                            ::  final_score_numerator
    real                            ::  final_score_denominator
    real                            ::  final_score

    type(ElectronDose)              ::  my_electron_dose

    integer                         ::  current_frame

    logical                         ::  input_shifts_are_in_angstroms

    integer                         ::  i,j
    real                            ::  x,y
    real                            ::  current_denominator
    real                            ::  current_critical_dose
    real                            ::  current_optimal_dose
    real                            ::  current_dose

    type(ContrastTransferFunction), allocatable ::  frame_ctf(:)

    logical,        parameter       ::  use_new_subroutine = .false.
    real,           parameter       ::  minimum_absolute_ctf_value = 0.001


    ! Initialise this_program with the program name and version number
    call this_program%Init('SumMovie', '1.0.2','2015')


    ! Do the user input
    call my_user_input%Init(this_program%program_name)


    user_supplied_input_filename   = my_user_input%GetFilenameFromUser('Input stack filename', &
                                   'The input file, containing your raw movie frames', &
                                   'INPUT_FILENAME', 'my_movie.mrc', .true.)

    ! Open the input file
    call input_imagefile%Init(user_supplied_input_filename%value)
    number_of_input_images = input_imagefile%GetStackSize()

    number_of_frames_per_movie      =   my_user_input%GetIntegerFromUser('Number of frames per movie', &
                                    'How many frames per micrograph?', &
                                    'number_of_frames_per_movie',IntegerToString(number_of_input_images), &
                                    min_value=1,max_value=number_of_input_images)

    user_supplied_output_filename   = my_user_input%GetFilenameFromUser('Output aligned sum file', &
                                   'The output file, containing a weighted sum of the aligned input frames', &
                                   'output_filename', 'my_aligned.mrc', .false.)

    user_supplied_shifts_input_filename   = my_user_input%GetFilenameFromUser('Input shifts file', &
                                   'This file will contain the computed X/Y shifts for each frame.', &
                                   'shifts_filename', 'my_shifts.txt', .true.)


    user_supplied_frc_output_filename   = my_user_input%GetFilenameFromUser('Output FRC file', &
                                   'This file will contain the computed FSC of the two half sums', &
                                   'frc_filename', 'my_frc.txt', .false.)

    user_supplied_first_frame      = my_user_input%GetIntegerFromUser('First frame to sum', &
                                  'A subset of frames is processed, starting from this frame', &
                                  'first_frame', '1', min_value = 1)

    user_supplied_last_frame       = my_user_input%GetIntegerFromUser('Last frame to sum', &
                                  'A subset of frames is processed, ending with this frame', 'LAST_FRAME', &
                                  IntegerToString(number_of_frames_per_movie%value), &
                                  min_value=user_supplied_first_frame%value, &
                                  max_value=number_of_frames_per_movie%value)


    pixel_size       = my_user_input%GetRealFromUser('Pixel size of images (A)', &
                                  'Pixel size of input images in Angstroms', 'PIXEL_SIZE', '1.0', min_value = 0.0e0)


    !user_supplied_apply_drift_filter   = my_user_input%GetLogicalFromUser('Apply drift filter?', &
    !                              'If yes, a corrective drift filter will be applied', 'USE_DRIFT_FILTER', 'NO')

    user_supplied_apply_drift_filter%value = .false.

    apply_dose_filter               =   my_user_input%GetLogicalFromUser('Apply dose filter?', &
                                    'Apply a dose-dependent filter to frames before summing them', &
                                    'apply_dose_filter','no')

    if (apply_dose_filter%value) then
    dose_per_frame                  =   my_user_input%GetRealFromUser('Dose per frame (e/A^2)', &
                                    'Dose per frame, in electrons per square Angstrom','dose_per_frame', &
                                    '1.0',min_value=0.00001e0)
    acceleration_voltage            =   my_user_input%GetRealFromUser('Acceleration voltage (kV)', &
                                    'Acceleration voltage during imaging','acceleration_voltage','300.0', &
                                    min_value=200.0e0,max_value=300.0e0)

    pre_exposure_amount         =   my_user_input%GetRealFromUser('Pre-exposure Amount(e/A^2)', &
                                        'Amount of pre-exposure prior to the first frame, in electrons per square Angstrom','pre_exposure_amount', &
                                        '0.0',min_value=0.0000)


    endif

    !apply_ctf                       =   my_user_input%GetLogicalFromUser('Apply CTF to frames?', &
    !                                    'Apply CTF to frames before aligning them','apply_ctf','no')

     apply_ctf%value = .false.

!    if (apply_ctf%value) then
 !       phase_flip_only             =   my_user_input%GetLogicalFromUser('Phase flip only?', &
 !                                       'Multiply by the sign of the CTF rather than its value','phase_flip_only','no')

 !       ctffind_results             =   my_user_input%GetFilenameFromUser('Results from CTFfind', &
 !                                       'Filename of CTFfind output','ctffind_results','my_ctf_results.txt',file_must_exist=.true.)
 !   endif

    if (apply_dose_filter%value) then
        restore_power      =       my_user_input%GetLogicalFromUser('Restore noise power after filtering?', &
                                    'Renormalise the summed image after filtering', &
                                    'restore_power','yes')
    else
        restore_power%value = .false.
    endif

!    if (apply_ctf%value) then
!        apply_ctf_wiener_filter     =   my_user_input%GetLogicalFromUser('Apply CTF Wiener-like filter', &
!                                        'Apply a Wiener-like filter (division by sum of squared CTF terms)'// &
!                                        ' to correct for effect of the CTF.','apply_ctf_wiener_filter','no')
 !       if (apply_ctf_wiener_filter%value) then
 !       ctf_wiener_filter_constant  =   my_user_input%GetRealFromUser('CTF Wiener constant', &
 !                                       'Constant to be added to sum of squared CTF terms in the denominator', &
 !                                       'ctf_wiener_filter_constant','0.1')
 !       endif
 !   else
        apply_ctf_wiener_filter%value = .false.
  !  endif


    call my_user_input%UpdateDefaults()
    write(*,*)

    ! Open the input stack and shifts file
    call my_shifts_file%Init(user_supplied_shifts_input_filename%value, OPEN_TO_READ)
    call my_frc_file%Init(user_supplied_frc_output_filename%value, OPEN_TO_WRITE, 2)

    ! How many movies in the stack?
    number_of_movies = number_of_input_images / number_of_frames_per_movie%value


    ! Checks
    if (number_of_frames_per_movie%value * number_of_movies .ne. number_of_input_images) then
        call this_program%TerminateWithFatalError('SumMovie','Number of images in input stack not '//&
                                                            'divisible by number of frames per movie')
    endif
    if (number_of_movies .ne. my_shifts_file%number_of_data_lines/2) then
        print *, 'number of data lines     = ', my_shifts_file%number_of_data_lines
        print *, 'number of data lines  /2 = ', my_shifts_file%number_of_data_lines/2
        print *, 'number of movies         = ', number_of_movies
        call this_program%TerminateWithFatalError('SumMovie','Number of lines in shifts file '//&
                                                            'does not match number of movies')
    endif


    number_of_frames_to_sum = (user_supplied_last_frame%value - user_supplied_first_frame%value) + 1

    ! Are the shifts in the file in Angstroms (default in recent versions) or pixels (original default)
    ! We check this by looking for "Shifts below are given in Angstroms" in the header
    input_shifts_are_in_angstroms = .not. StringIsBlank(grep(user_supplied_shifts_input_filename%value, &
                                                             'Shifts below are given in Angstroms'))
    if (input_shifts_are_in_angstroms) then
        write(*,'(a)') 'Input frame shifts assumed to be in Angstroms'
    else
        write(*,'(a)') 'Input frame shifts assumed to be in pixels'
    endif


    ! Allocation
    if (use_new_subroutine) then
        allocate(image_stack(number_of_frames_per_movie%value))
    else
        allocate(image_stack(number_of_frames_to_sum))
    endif
    allocate(total_x_shifts(number_of_frames_per_movie%value))
    allocate(total_y_shifts(number_of_frames_per_movie%value))


    ! Electron dose parameters
    if (apply_dose_filter%value) then
        call my_electron_dose%Init(acceleration_voltage%value)
    endif

    ! Parse CTFfind results
  !  if (apply_ctf%value) then
  !      call ParseCTFFindResultsFile(ctffind_results%value,pixel_size%value,frame_ctf)
  !      if (size(frame_ctf) .ne. number_of_frames_per_movie%value) then
  !          call this_program%TerminateWithFatalError('Unblur',&
  !                                  'Number of CTF parameters = '//IntegerToString(size(frame_ctf))// &
  !                                  'Number of frames = '//IntegerToString(number_of_frames_per_movie%value))
  !      endif
  !  endif

    ! Loop over movies
    do movie_counter=1,number_of_movies
        write(*,'(//2(a,i0),a//)') 'Summing frames of micrograph ', movie_counter, ' of ', number_of_movies, '...'

        ! Read the shifts
        call my_shifts_file%ReadNextDataLine(total_x_shifts)
        call my_shifts_file%ReadNextDataLine(total_y_shifts)

        ! Convert shifts from Angstroms to pixels
        if (input_shifts_are_in_angstroms) then
            total_x_shifts = total_x_shifts / pixel_size%value
            total_y_shifts = total_y_shifts / pixel_size%value
        endif

        if (use_new_subroutine) then

            ! Read the images
            write(*,'(a)') 'Reading images...'
            call my_progress_bar%Begin(number_of_frames_per_movie%value)
            do image_counter=1,number_of_frames_per_movie%value
                call image_stack(image_counter)%ReadFromDisk(input_imagefile, &
                                                              (movie_counter-1)*number_of_frames_per_movie%value &
                                                             +image_counter)
                call my_progress_bar%Update(image_counter)
            enddo
            call my_progress_bar%Finish()


            ! Write header comments to FRC numeric text file
            if (movie_counter .eq. 1) then
                call my_frc_file%WriteCommentLine('SumMovie FRC for input stack : '//user_supplied_input_filename%value)
                call my_frc_file%WriteCommentLine('Number of micrographs: '//IntegerToString(number_of_movies))
                call my_frc_file%WriteCommentLine('Number of frames per movie: '//&
                                                   IntegerToString(number_of_frames_per_movie%value))
                call my_frc_file%WriteCommentLine('Pixel size (A): '//RealToString(pixel_size%value,4))
                call my_frc_file%WriteCommentLine('  1/(A)          FRC')
                call my_frc_file%WriteCommentLine('-----------------------')
                call my_frc_file%WriteCommentLine('')
            endif

            ! Call subroutine to do the actual summing
            call sum_movie( image_stack,sum_image,pixel_size%value,                         &
                            user_supplied_first_frame%value,user_supplied_last_frame%value, &
                            apply_shifts=.true.,                                            &
                            apply_drift_filter=user_supplied_apply_drift_filter%value,      &
                            apply_dose_filter=apply_dose_filter%value,                      &
                            apply_ctf=apply_ctf%value,                                      &
                            ctf_already_applied=.false.,                                    &
                            phase_flip_only=phase_flip_only%value,                          &
                            restore_power=restore_power%value,                              &
                            ctf_wiener_filter=apply_ctf_wiener_filter%value,                &
                            x_shifts=total_x_shifts,                                        &
                            y_shifts=total_y_shifts,                                        &
                            ctf=frame_ctf,                                                  &
                            minimum_absolute_ctf_value=minimum_absolute_ctf_value,          &
                            electron_dose=my_electron_dose,                                 &
                            exposure_per_frame=dose_per_frame%value,                        &
		            pre_exposure_amount=pre_exposure_amount%value,                  &
                            frc_file=my_frc_file,                                           &
                            movie_number=movie_counter,                                     &
                            score=final_score,                                              &
                            ctf_wiener_filter_constant=ctf_wiener_filter_constant%value     &
                            )

        else

            if (apply_ctf%value) then
               call this_program%TerminateWithFatalError('sum_movie','Need to use new subroutine to apply ctf')
            endif

            ! Read the images
            write(*,'(a)') 'Reading images...'
            call my_progress_bar%Begin(number_of_frames_to_sum)
            do image_counter=1,number_of_frames_to_sum
                call image_stack(image_counter)%ReadFromDisk(input_imagefile, &
                                                              (movie_counter-1)*number_of_frames_per_movie%value &
                                                             +user_supplied_first_frame%value+image_counter-1)
                call my_progress_bar%Update(image_counter)
            enddo
            call my_progress_bar%Finish()


            ! FFT
            write(*,'(a)') 'Fourier transforming images...'
            call my_progress_bar%Begin(number_of_frames_to_sum)
            progress_counter = 0
            !$omp parallel default(shared) private(image_counter)
            !$omp do
            do image_counter =1,number_of_frames_to_sum
                call image_stack(image_counter)%ForwardFFT()
                !$omp atomic
                progress_counter = progress_counter + 1
                call my_progress_bar%Update(progress_counter)
            enddo
            !$omp enddo
            !$omp end parallel
            call my_progress_bar%Finish()

            ! Shift images
            if (apply_dose_filter%value) then
                write(*,'(a)') 'Shifting images & applying dose filter...'
            else
                write(*,'(a)') 'Shifting images...'
            endif
            call my_progress_bar%Begin(number_of_frames_to_sum)
            progress_counter = 0
            !$omp parallel default(shared) private(image_counter)
            !$omp do
            do current_frame=user_supplied_first_frame%value, user_supplied_last_frame%value
                image_counter = current_frame - user_supplied_first_frame%value + 1
                call image_stack(image_counter)%PhaseShift(total_x_shifts(current_frame), &
                                                           total_y_shifts(current_frame), &
                                                           0.0e0)

                if (user_supplied_apply_drift_filter%value) then
                    ! What is the estimated drift within this frame?
                    if (current_frame .eq. 1) then
                        current_drift_x = total_x_shifts(2) - total_x_shifts(1)
                        current_drift_y = total_y_shifts(2) - total_y_shifts(1)
                    else if (current_frame .eq. number_of_frames_per_movie%value) then
                        current_drift_x = total_x_shifts(number_of_frames_per_movie%value) &
                                        - total_x_shifts(number_of_frames_per_movie%value - 1)
                        current_drift_y = total_y_shifts(number_of_frames_per_movie%value) &
                                        - total_y_shifts(number_of_frames_per_movie%value - 1)
                    else
                        current_drift_x = 0.5e0 * (   total_x_shifts(current_frame+1) &
                                                    - total_x_shifts(current_frame-1))
                        current_drift_y = 0.5e0 * (   total_y_shifts(current_frame+1) &
                                                    - total_y_shifts(current_frame-1))
                    endif
                    call image_stack(image_counter)%ApplyDriftFilter(current_drift_x, current_drift_y, 0.0e0)
                endif
                if (apply_dose_filter%value) then
                    !
                    call my_electron_dose%ApplyDoseFilterToImage(image_stack(image_counter), &
                                                                 dose_start=((current_frame-1)*dose_per_frame%value) + pre_exposure_amount%value, &
                                                                 dose_finish=(current_frame*dose_per_frame%value) + pre_exposure_amount%value, &
                                                                 pixel_size=pixel_size%value)
                endif
                !$omp atomic
                progress_counter = progress_counter + 1
                call my_progress_bar%Update(progress_counter)
            enddo
            !$omp enddo
            !$omp end parallel
            call my_progress_bar%Finish()

            ! Compute two sums from halves of the stack and do an FRC
            call sum_image%Allocate(mould=image_stack(1))
            sum_image = (0e0, 0e0)

            call sum_image_2%Allocate(mould=sum_image)
            sum_image_2 = (0e0, 0e0)

            do image_counter=1, number_of_frames_to_sum
                if (IsOdd(image_counter)) then
                    call sum_image  %AddImage(image_stack(image_counter))
                else
                    call sum_image_2%AddImage(image_stack(image_counter))
                endif
            enddo

            if (number_of_frames_to_sum .gt. 1) then
                fsc_curve = sum_image%GetFSCWith(sum_image_2, 100, pixel_size%value)
                call sum_image%AddImage(sum_image_2)
                call fsc_curve%CopyXData(frc_x_data)
                call fsc_curve%CopyYData(frc_y_data)

                ! Work out the Final score as per original unblur
                final_score_numerator = 0.0e0
                final_score_denominator = 0.0e0
                do bin_counter = 1, 100
                    if (frc_x_data(bin_counter) .le. 0.05e0) then
                        final_score_numerator = final_score_numerator + frc_y_data(bin_counter) * real(bin_counter - 1)
                        final_score_denominator = final_score_denominator + real(bin_counter - 1)
                    endif
                enddo
                final_score = final_score_numerator / final_score_denominator

                ! write out the fsc_curve.
                if (movie_counter .eq. 1) then
                    call my_frc_file%WriteCommentLine('SumMovie FRC for input stack : '//user_supplied_input_filename%value)
                    call my_frc_file%WriteCommentLine('Number of micrographs: '//IntegerToString(number_of_movies))
                    call my_frc_file%WriteCommentLine('Number of frames per movie: '//&
                                                       IntegerToString(number_of_frames_per_movie%value))
                    call my_frc_file%WriteCommentLine('Pixel size (A): '//RealToString(pixel_size%value,4))
                    call my_frc_file%WriteCommentLine('  1/(A)          FRC')
                    call my_frc_file%WriteCommentLine('-----------------------')
                    call my_frc_file%WriteCommentLine('')
                endif
                call my_frc_file%WriteCommentLine('Micrograph '//IntegerToString(movie_counter)//' of '//&
                                                  IntegerToString(number_of_movies)//'. Score: '//RealToString(final_score,3))
                do bin_counter = 1, 100
                    temp_real(1) = frc_x_data(bin_counter)
                    temp_real(2) = frc_y_data(bin_counter)

                    call my_frc_file%WriteDataLine(temp_real)
                enddo
            else
                final_score = 1.0e0
            endif


           ! If necessary, restore the power

           if (apply_dose_filter%value .and. restore_power%value) then

                !$omp parallel default(shared) private(j, y, i, x, image_counter, current_denominator, current_critical_dose, current_dose, current_optimal_dose)
                !$omp do

                ! for this pixel work out what to divide by. first we need the radius

                do j=1, sum_image%GetLogicalDimension(2)
                    y = (sum_image%LogicalIndexGivenPhysicalIndexInFourierSpace(j,2) * sum_image%fourier_voxel_size(2))**2
                    do i=1, sum_image%physical_upper_bound_complex(1)
                        x = ((i-1) * sum_image%fourier_voxel_size(1))**2
                        !
                        if (i .eq. 1 .and. j .eq. 1) then
                            current_critical_dose = critical_dose_at_dc
                        else
                            current_critical_dose = my_electron_dose%CriticalDose(sqrt(x+y) / pixel_size%value)
                        endif


                        current_optimal_dose = my_electron_dose%OptimalDoseGivenCriticalDose(current_critical_dose)

                        ! add up for all frames

                        current_denominator = 0.0e0

                        do image_counter = 1, number_of_frames_per_movie%value

                            current_dose = (real(image_counter) * dose_per_frame%value) + pre_exposure_amount%value

                            if (current_dose .lt. current_optimal_dose) then
                                current_denominator = current_denominator &
                                                    + my_electron_dose%DoseFilter(current_dose, current_critical_dose)**2
                            endif
                        enddo



                        if (current_denominator .gt. 0.0) then
                            current_denominator = sqrt(current_denominator)
                            sum_image%complex_values(i, j, 1) = sum_image%complex_values(i, j, 1) / current_denominator
                        endif

                    enddo
                  enddo

                !$omp enddo
                !$omp end parallel

           endif
        endif    ! end of check for new subroutine

        ! Write out the sum
        call sum_image%WriteToDisk(user_supplied_output_filename%value, movie_counter)


    enddo ! end of loop over movies

    ! Deallocation
    if (allocated(image_stack)) deallocate(image_stack)
    if (allocated(total_x_shifts)) deallocate(total_x_shifts)
    if (allocated(total_y_shifts)) deallocate(total_y_shifts)

    call this_program%Terminate()


end program unblur
