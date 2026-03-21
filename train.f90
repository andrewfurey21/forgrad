
module forgrad
        use, intrinsic :: iso_fortran_env, only: f32 => real32
        implicit none(type, external)
        private
        public f32, read_image_data, generate_random_indices
        public init_grads, init_weights, one_hot_encoding
        public relu_forwards, relu_backwards, sigmoid_forwards, sigmoid_backwards, identity_forwards, identity_backwards
        public linear_forwards, linear_backwards, linear_gradient
        public mse_forwards, mse_backwards


        integer, parameter :: max_line_length = 4000
contains
        subroutine parse_line_csv(line, img_size, image, answer)
                character(len=max_line_length), intent(in) :: line
                integer, intent(in) :: img_size

                real(f32), dimension(1, img_size), intent(out) :: image
                real(f32), dimension(1), intent(out) :: answer

                character :: answer_ascii, current_char
                integer :: char_as_number = 0
                integer :: line_index
                integer :: pixel_index = 1
                real(f32) :: number

                ! some assumptions about the data are made, no spaces, comma seperated. answer is one digit.
                answer_ascii = line(1:1)
                answer(1:1) = iachar(answer_ascii) - iachar('0')

                pixel_index = 1
                char_as_number = 0
                number = 0

                do line_index = 2, max_line_length
                        current_char = line(line_index:line_index)
                        if (current_char == " ") then
                                image(1:1, pixel_index:pixel_index) = char_as_number
                                exit
                        else if (current_char == ",") then
                                if (line_index == 2) then
                                        cycle
                                end if
                                image(1:1, pixel_index:pixel_index) = real(char_as_number, kind=f32) / 255.0
                                char_as_number = 0
                                pixel_index = pixel_index + 1
                        else
                                char_as_number = char_as_number * 10 + (iachar(current_char) - iachar('0'))
                        end if
                end do
        end subroutine

        subroutine read_image_data(path, num_imgs, img_size, images, answers)
                character(len=*), intent(in) :: path
                integer, intent(in) :: num_imgs, img_size

                real(f32), dimension(num_imgs, img_size), intent(out) :: images
                real(f32), dimension(num_imgs), intent(out) :: answers

                integer :: file, iostat, current_line
                character(len=max_line_length) :: line
                logical :: path_exists = .false.

                inquire(file=path, exist=path_exists)
                if (.not. path_exists) then
                        error stop ": could not find " // path
                end if
                open(newunit=file, file=path, status="old", action="read")

                do current_line = 1, num_imgs
                        read(file, "(A)", iostat=iostat) line
                        line = trim(line)
                        call parse_line_csv(line, img_size, images(current_line:current_line, :), answers(current_line:current_line))
                end do

                close(file)
        end subroutine

        subroutine generate_random_indices(num_indices, indices)
                integer, intent(in) :: num_indices
                integer, dimension(num_indices), intent(out) :: indices
                integer :: i, chosen, temp
                real :: n

                call random_seed()

                do i = 1, num_indices
                        indices(i:i) = i
                end do

                do i = num_indices, 1, -1
                        call random_number(n)
                        chosen = floor(n * i) + 1
                        temp = indices(chosen)
                        indices(chosen) = indices(i)
                        indices(i) = temp
                end do

                ! call check_indices(indices)
        end subroutine

        subroutine check_indices(indices)
                integer, intent(in) :: indices(:)
                integer :: i, j

                do i = 1, size(indices)
                        if (indices(i) > size(indices)) then
                                print *, "index is too big: ", i, indices(i)
                                error stop
                        end if

                        do j = (i + 1), size(indices)
                                if (indices(i) == indices(j)) then
                                        error stop " this is not a unique array!"
                                end if
                        end do
                end do
        end subroutine

        subroutine init_weights(weights, rows, cols)
                real(f32), intent(out) :: weights(rows, cols)
                integer, intent(in) :: rows, cols

                integer :: i, j
                real(f32) :: n

                call random_seed()

                do i = 1, rows
                        do j = 1, cols
                                call random_number(n)
                                weights(i, j) = n * 2 - 1
                        end do
                end do
        end subroutine

        subroutine init_grads(grads, rows, cols)
                real(f32), intent(out) :: grads(rows, cols)
                integer, intent(in) :: rows, cols
                grads = 0
        end subroutine

        function one_hot_encoding(answers, batch_size, num_answers) result(one_hot)
                integer :: num_answers, batch_size, i
                real(f32) :: answers(batch_size)
                real(f32) :: one_hot(batch_size, num_answers)
                one_hot = 0

                ! one_hot(:, int(answers + 1)) = 1
                do i = 1, batch_size
                        one_hot(i, int(answers(i) + 1)) = 1
                end do
        end function

        subroutine relu_forwards(inputs, outputs)
                real(f32), intent(in) :: inputs(:, :)
                real(f32), intent(out) :: outputs(:, :)
                outputs = max(0.0, inputs)
        end subroutine

        subroutine relu_backwards(outputs, grads)
                real(f32), intent(in) :: outputs(:, :)
                real(f32), intent(out) :: grads(:, :)
                grads = merge(1.0, 0.0, outputs > 0)
        end subroutine

        subroutine identity_forwards(inputs, outputs)
                real(f32), intent(in) :: inputs(:, :)
                real(f32), intent(out) :: outputs(:, :)
                outputs = inputs
        end subroutine

        subroutine identity_backwards(outputs, grads)
                real(f32), intent(in) :: outputs(:, :)
                real(f32), intent(out) :: grads(:, :)
                grads = outputs
        end subroutine

        subroutine sigmoid_forwards(inputs, outputs)
                real(f32), intent(in) :: inputs(:, :)
                real(f32), intent(out) :: outputs(:, :)
                outputs = 1. / (1. + exp(-inputs))
        end subroutine

        subroutine sigmoid_backwards(outputs, grads)
                real(f32), intent(in) :: outputs(:, :)
                real(f32), intent(out) :: grads(:, :)
                ! grads = merge(1.0, 0.0, outputs > 0)
                grads = 1. / (1. + exp(-outputs)) * (1. - 1. / (1. + exp(-outputs)))
        end subroutine

        subroutine linear_forwards(inputs, layer, outputs)
                real(f32), intent(in) :: inputs(:, :)
                real(f32), intent(in) :: layer(:, :)
                real(f32), intent(out) :: outputs(:, :)
                outputs = matmul(inputs, layer)
        end subroutine

        subroutine linear_backwards(dlossdoutput, layer, dlossdinput)
                real(f32), intent(in) :: dlossdoutput(:, :)
                real(f32), intent(in) :: layer(:, :)
                real(f32), intent(out) :: dlossdinput(:, :)
                dlossdinput = matmul(dlossdoutput, transpose(layer))
        end subroutine

        subroutine linear_gradient(inputs, dlossdoutput, grad)
                real(f32), intent(in) :: inputs(:, :)
                real(f32), intent(in) :: dlossdoutput(:, :)
                real(f32), intent(out) :: grad(:, :)

                grad = matmul(transpose(inputs), dlossdoutput)
        end subroutine

        subroutine mse_forwards(answers, expected, batch_size, loss)
                integer, intent(in) :: batch_size
                real(f32), intent(in) :: answers(:, :) ! (batch size, 10)
                real(f32), intent(in) :: expected(:, :)
                real(f32), intent(out) :: loss

                loss = sum((expected - answers) ** 2) / batch_size
        end subroutine

        subroutine mse_backwards(answers, expected, batch_size, dloss)
                integer, intent(in) :: batch_size
                real(f32), intent(in) :: answers(:, :) ! (batch size, 10)
                real(f32), intent(in) :: expected(:, :)
                real(f32), intent(out) :: dloss(:, :)

                dloss = 2 * (expected - answers) / batch_size
        end subroutine

end module forgrad

program train
        use forgrad
        implicit none(type, external)

        integer, parameter :: img_size = 28 * 28
        integer, parameter :: batch_size = 128
        integer, parameter :: num_epochs = 10


        integer, parameter :: num_train_images = 60000
        character(len=:), allocatable :: training_data_csv_path
        real(f32), dimension(num_train_images, img_size) :: training_images
        real(f32), dimension(num_train_images) :: training_answers
        integer, dimension(num_train_images) :: training_indices

        real(f32), dimension(batch_size, img_size) :: batch_images
        real(f32), dimension(batch_size) :: batch_answers
        real(f32), dimension(batch_size, 10) :: batch_one_hot_answers
        integer, dimension(batch_size) :: batch_indices

        integer :: epoch, batch

        ! model
        real(f32), dimension(img_size, 128) :: layer_1
        real(f32), dimension(128, 64) :: layer_2
        real(f32), dimension(64, 10) :: layer_3

        real(f32), dimension(batch_size, 128) :: layer_1_outputs
        real(f32), dimension(batch_size, 64) :: layer_2_outputs
        real(f32), dimension(batch_size, 10) :: layer_3_outputs

        real(f32), dimension(batch_size, 128) :: layer_1_act
        real(f32), dimension(batch_size, 64) :: layer_2_act
        real(f32), dimension(batch_size, 10) :: layer_3_act

        real(f32), dimension(img_size, 128) :: layer_1_grads
        real(f32), dimension(128, 64) :: layer_2_grads
        real(f32), dimension(64, 10) :: layer_3_grads

        real(f32), dimension(batch_size, 128) :: dlossdlayer1
        real(f32), dimension(batch_size, 64) :: dlossdlayer2
        real(f32), dimension(batch_size, 10) :: dlossdlayer3

        real(f32), dimension(batch_size, 128) :: dlossdlayer1act
        real(f32), dimension(batch_size, 64) :: dlossdlayer2act
        real(f32), dimension(batch_size, 10) :: dlossdlayer3act

        real(f32) :: loss, learning_rate, total_loss

        learning_rate = 0.00001
        total_loss = 0
        loss = 0

        training_data_csv_path = "./data/mnist_train.csv"
        print *, "Reading training data: " // training_data_csv_path // "..."
        call read_image_data(training_data_csv_path, num_train_images, img_size, training_images, training_answers)
        print *, "done."

        call init_grads(layer_1_grads, img_size, 128)
        call init_grads(layer_2_grads, 128, 64)
        call init_grads(layer_3_grads, 64, 10)

        call init_weights(layer_1, img_size, 128)
        call init_weights(layer_2, 128, 64)
        call init_weights(layer_3, 64, 10)

        do epoch = 1, num_epochs
                print *, "Epoch ", epoch
                call generate_random_indices(num_train_images, training_indices)
                do batch = 1, num_train_images - batch_size - 1, batch_size
                ! do batch = 1, 10, batch_size
                        batch_indices = training_indices(batch:(batch + batch_size - 1))
                        batch_answers = training_answers(batch_indices(1:batch))
                        batch_one_hot_answers = one_hot_encoding(batch_answers, batch_size, 10)
                        batch_images = training_images(batch_indices, :)

                        call linear_forwards(batch_images, layer_1, layer_1_outputs)
                        call relu_forwards(layer_1_outputs, layer_1_act)

                        call linear_forwards(layer_1_outputs, layer_2, layer_2_outputs)
                        call relu_forwards(layer_2_outputs, layer_2_act)

                        call linear_forwards(layer_2_outputs, layer_3, layer_3_outputs)
                        ! call identity_forwards(layer_3_outputs, layer_3_act)
                        call sigmoid_forwards(layer_3_outputs, layer_3_act)

                        call mse_forwards(layer_3_act, batch_one_hot_answers, batch_size, loss)

                        ! print *, "Loss: ", loss
                        total_loss = total_loss + loss

                        call mse_backwards(layer_3_act, batch_one_hot_answers, batch_size, dlossdlayer3act)

                        ! call identity_backwards(dlossdlayer3act, dlossdlayer3)
                        call sigmoid_backwards(dlossdlayer3act, dlossdlayer3)
                        call linear_backwards(dlossdlayer3, layer_3, dlossdlayer2act)

                        call relu_backwards(dlossdlayer2act, dlossdlayer2)
                        call linear_backwards(dlossdlayer2, layer_2, dlossdlayer1act)

                        call relu_backwards(dlossdlayer1act, dlossdlayer1)

                        call linear_gradient(batch_images, dlossdlayer1 * loss, layer_1_grads)
                        call linear_gradient(batch_images, dlossdlayer2 * loss, layer_2_grads)
                        call linear_gradient(batch_images, dlossdlayer3 * loss, layer_3_grads)

                        layer_1 = layer_1 - layer_1_grads * learning_rate
                        layer_2 = layer_2 - layer_2_grads * learning_rate
                        layer_3 = layer_3 - layer_3_grads * learning_rate
                end do

                print *, "Loss: ", total_loss / (num_train_images / batch_size)
                total_loss = 0
        end do
end program train
