function main()

    % Define colors used in the GUI
    COLOR1 = [1.00, 1.00, 1.00];
    COLOR2 = [0.64, 0.08, 0.18];
    COLOR3 = [0.94, 0.90, 0.84];
    COLOR4 = [0.96, 0.75, 0.59];
    COLOR5 = [0.53, 0.32, 0.20];
    COLOR6 = [0.60, 0.60, 0.60];

    % Define a default image path
    imageFile = "pics/abdalal-small.jpg";


    % The following few lines initialize the GUI
    f = figure(Position=[360,500,450,285], Units="normalized", Color=COLOR3);
    movegui(f, "center");

    oldImageFrame = axes(Units="pixels", Position=[20, 160, 200, 100]); 
    newImageFrame = axes(Units="pixels", Position=[20, 40, 200, 100]); 
    
    slider1 = uicontrol(Style="slider", Position=[250, 220, 120, 5], Units="normalized", Callback={@runDetection}, Min=100, Max=2000, Value=1000, BackgroundColor=COLOR4);
    slider2 = uicontrol(Style="slider", Position=[250, 200, 120, 5], Units="normalized", Callback={@runDetection}, Min=1, Max=6, Value=3, BackgroundColor=COLOR4);
    slider3 = uicontrol(Style="slider", Position=[250, 180, 120, 5], Units="normalized", Callback={@runDetection}, Min=0.05, Max=1, Value=0.65, BackgroundColor=COLOR4);
    slider4 = uicontrol(Style="slider", Position=[250, 160, 120, 5], Units="normalized", Callback={@runDetection}, Min=0, Max=2, Value=0.1, BackgroundColor=COLOR4);
    slider5 = uicontrol(Style="slider", Position=[250, 140, 120, 5], Units="normalized", Callback={@runDetection}, Min=0, Max=2, Value=1, BackgroundColor=COLOR4);
    slider6 = uicontrol(Style="slider", Position=[250, 120, 120, 5], Units="normalized", Callback={@runDetection}, Min=1, Max=20, Value=15, BackgroundColor=COLOR4);
    
    annotation("textbox", Position=[0.5, 0.9, 0.1, 0.1], String="Facial Feature Detection", HorizontalAlignment="center", VerticalAlignment="middle", Color=COLOR1, BackgroundColor=COLOR5, EdgeColor=COLOR6, LineWidth=2, FontSize=30, FontWeight="bold");
    uicontrol(Style="text", Position=[375, 220, 100, 5], Units="normalized", String="Resolution Multiplier", BackgroundColor=COLOR3, ForegroundColor=COLOR2, FontSize=10);
    uicontrol(Style="text", Position=[375, 200, 100, 5], Units="normalized", String="Gaussian Kernel Multiplier", BackgroundColor=COLOR3, ForegroundColor=COLOR2, FontSize=10);
    uicontrol(Style="text", Position=[375, 180, 100, 5], Units="normalized", String="Scaling Multiplier", BackgroundColor=COLOR3, ForegroundColor=COLOR2, FontSize=10);
    uicontrol(Style="text", Position=[375, 160, 100, 5], Units="normalized", String="Intensity Parameter (Lighten)", BackgroundColor=COLOR3, ForegroundColor=COLOR2, FontSize=10);
    uicontrol(Style="text", Position=[375, 140, 100, 5], Units="normalized", String="Intensity Parameter (Darken)", BackgroundColor=COLOR3, ForegroundColor=COLOR2, FontSize=10);
    uicontrol(Style="text", Position=[375, 120, 100, 5], Units="normalized", String="Face Detection Threshold", BackgroundColor=COLOR3, ForegroundColor=COLOR2, FontSize=10);
    uicontrol(Style="pushbutton", Position=[250, 240, 120, 10], Units="normalized", String="Select Image", Callback={@selectImage}, BackgroundColor=COLOR5, ForegroundColor=COLOR1, FontSize=14);

    rotateCheckbox = uicontrol(Style="checkbox", Position=[325, 100, 45, 10], Units="normalized", String="Detect Titlted Faces", Callback={@runDetection}, Value=false,  BackgroundColor=COLOR5, ForegroundColor=COLOR1);
    fallbackCheckbox = uicontrol(Style="checkbox", Position=[250, 100, 45, 10], Units="normalized", String="Enable Fallback", Callback={@runDetection}, Value=false,  BackgroundColor=COLOR5, ForegroundColor=COLOR1);
    faceCheckbox = uicontrol(Style="checkbox", Position=[250, 80, 45, 10], Units="normalized", String="Highlight Face", Callback={@runDetection}, Value=false,  BackgroundColor=COLOR5, ForegroundColor=COLOR1);
    mouthCheckbox = uicontrol(Style="checkbox", Position=[325, 80, 45, 10], Units="normalized", String="Highlight Mouth", Callback={@runDetection}, Value=true,  BackgroundColor=COLOR5, ForegroundColor=COLOR1);
    noseCheckbox = uicontrol(Style="checkbox", Position=[250, 60, 45, 10], Units="normalized", String="Highlight Nose", Callback={@runDetection}, Value=true,  BackgroundColor=COLOR5, ForegroundColor=COLOR1);
    eyesCheckbox = uicontrol(Style="checkbox", Position=[325, 60, 45, 10], Units="normalized", String="Highlight Eyes", Callback={@runDetection}, Value=true,  BackgroundColor=COLOR5, ForegroundColor=COLOR1);

    oldImageFrame.Units = "normalized";
    newImageFrame.Units = "normalized";

    % Runs the dection
    runDetection;

    function faceDetect(imageFile, fallback, face, eyes, nose, mouth, rotate, defaultSize, defaultGamma, defaultScale, defaultLighten, defaultDarken, faceThreshold)
        
        % Define a couple of inline functions used later in the detection
        calculateArea = @(bbox) bbox(:,3) .* bbox(:,4); % This returns the area of the given [x, y, height, width] object  
        calculateProportion = @(rect, faceArea) (rect(3) * rect(4)) / faceArea; % This returns the ratio of the facial feature to the face
        
        originalImage = readImageAsRGB(imageFile); % Reads all images as RGB images so that they are easier to work with
        originalImage = downScaleLargeImage(originalImage); % Downscale any image larger than 1920x1080, to load larger images faster
        
        % The next few lines define dynamic variables that are set in relation to the size of the image
        [height, width, ~] = size(originalImage);
        multipler = sqrt(height * width) / defaultSize;
        gamma = multipler * defaultGamma;
        scale = (multipler + 1) * defaultScale;
        lineWidth = max(1, round(4 * multipler));

        % Do some preprocessing, check the usser manual for info on the techniques used
        processedImage = processImage(originalImage, gamma, scale, defaultLighten, defaultDarken);
        displayProcessedImage = processedImage;
        
        displayImage = originalImage;
        
        % Use Viola-Jones to detct faces in the image
        faceDetector = vision.CascadeObjectDetector(MergeThreshold=faceThreshold);
        faces = step(faceDetector, processedImage);

        % The following if statement is for the "Detect Titled Faces" checkbox
        % It basically rotates the image 22.5 degrees (both directions) unitl it finds a face(s)
        % And then the feature detection can be done on the detected faces of the rotated image
        angle = 0;
        if isempty(faces) && rotate
            [faces, rotatedImage, correctAngle] = detectTiltedFaces(processedImage, faceDetector);
            if ~isempty(faces)
                angle = correctAngle;
                processedImage = rotatedImage;
                displayImage = imrotate(displayImage, angle);
            end
        end

        for i = 1:size(faces,1) % Run do the detection on every detected face
            faceRect = faces(i, :);
            displayFaceRect = faceRect/scale;
            faceArea = displayFaceRect(3) * displayFaceRect(4);
            
            if face
                % Draw a box around the face if the "Highlight Face" checkbox is checked
                displayImage = insertObjectAnnotation(displayImage, "rectangle", displayFaceRect, ['Face #', num2str(i)], LineWidth=lineWidth, Color="Red", FontColor="White", FontSize=ceil(8+(sqrt(faceArea)/10)));
            end
            
            if eyes % Only detect the eyes if the "Highlight Eyes" checkbox is checked
                upperHalfFace = faceRect;
                
                % Only look for the eyes in the upper half of the face, (helps with accuracy and performance)
                upperHalfFace(4) = upperHalfFace(4)/(1+(2/3));
                upperCroppedFace = imcrop(processedImage, upperHalfFace);
                eyeDetector = vision.CascadeObjectDetector("EyePairBig", MergeThreshold=5);
                eyeRegion = step(eyeDetector, upperCroppedFace);
                
                % Lower the threshold until some eyes are found or the threshold is at 1
                while isempty(eyeRegion) && eyeDetector.MergeThreshold > 1
                    eyeDetector.MergeThreshold = eyeDetector.MergeThreshold -1;
                    eyeRegion = step(eyeDetector, upperCroppedFace);
                end
                
                if ~isempty(eyeRegion)
                    newEyeRegion = [];

                    % Checks if the ratio of eyes is reasonable
                    for j = 1:size(eyeRegion,1)
                        if calculateProportion(eyeRegion(j, :), faceArea) > 0.05
                            newEyeRegion = [newEyeRegion, eyeRegion(j, :)]; %#ok<AGROW>
                        end
                    end
                    if ~isempty(newEyeRegion)
                        eyeRegion = newEyeRegion;
                    end
                    
                    % Picks the biggest pair detected
                    [~, prominentIdx] = max(calculateArea(eyeRegion));
                    eyeRect = eyeRegion(prominentIdx, :);
                
                    % Makes the box larger by 10%, to be certain it covers the correct area
                    scaleFactor = 1.1;
                    newWidth = eyeRect(3) * scaleFactor;
                    newHeight = eyeRect(4) * scaleFactor;
                
                    deltaX = (newWidth - eyeRect(3)) / 2;
                    deltaY = (newHeight - eyeRect(4)) / 2;
                    newX = eyeRect(1) - deltaX;
                    newY = eyeRect(2) - deltaY;
                
                    eyeRect = [newX, newY, newWidth, newHeight];
                
                    eyeRect(1:2) = eyeRect(1:2) + upperHalfFace(1:2);
                    eyeRect = eyeRect/scale;
                    
                    % Draw a box around the detected pair
                    displayImage = insertShape(displayImage, "Rectangle", eyeRect, LineWidth=lineWidth, Color="Blue");
                
                else % In case pair detection fails (maybe one eye is covered?), do the same steps previously but on each eye separately
                    % Only difference here is that it only checks the left/right halfs of the upper face for the left/right eyes respectively

                    [upperHeight, upperWidth, ~] = size(upperCroppedFace);
                    upperLeftRect = [1, 1, upperWidth/(1+(9/11)), upperHeight];
                    upperLeftCroppedFace = imcrop(upperCroppedFace, upperLeftRect);
                
                    leftEyeDetector = vision.CascadeObjectDetector("LeftEye", MergeThreshold=10);
                    leftEyeRegion = step(leftEyeDetector, upperLeftCroppedFace);
                    while isempty(leftEyeRegion) && leftEyeDetector.MergeThreshold > 1
                        leftEyeDetector.MergeThreshold = leftEyeDetector.MergeThreshold -1;
                        leftEyeRegion = step(leftEyeDetector, upperLeftCroppedFace);
                    end

                    if ~isempty(leftEyeRegion)
                        [~, prominentIdx] = max(calculateArea(leftEyeRegion));
                        leftEyeRect = leftEyeRegion(prominentIdx, :);
                
                        scaleFactor = 1.1;
                        newWidth = leftEyeRect(3) * scaleFactor;
                        newHeight = leftEyeRect(4) * scaleFactor;
                
                        deltaX = (newWidth - leftEyeRect(3)) / 2;
                        deltaY = (newHeight - leftEyeRect(4)) / 2;
                        newX = leftEyeRect(1) - deltaX;
                        newY = leftEyeRect(2) - deltaY;
                
                        leftEyeRect = [newX, newY, newWidth, newHeight];
                
                        leftEyeRect(1:2) = leftEyeRect(1:2) + upperHalfFace(1:2);
                        leftEyeRect = leftEyeRect/scale;
                        displayImage = insertShape(displayImage, "Rectangle", leftEyeRect, LineWidth=lineWidth, Color="Blue");
                    end
                
                    [upperHeight, upperWidth, ~] = size(upperCroppedFace);
                    upperRightRect = [upperWidth/(2+(2/9)), 1, upperWidth/(1+(9/11)), upperHeight];
                    upperRightCroppedFace = imcrop(upperCroppedFace, upperRightRect);
                
                    rightEyeDetector = vision.CascadeObjectDetector("RightEye", MergeThreshold=10);
                    rightEyeRegion = step(rightEyeDetector, upperRightCroppedFace);
                    while isempty(rightEyeRegion) && rightEyeDetector.MergeThreshold > 1
                        rightEyeDetector.MergeThreshold = rightEyeDetector.MergeThreshold -1;
                        rightEyeRegion = step(rightEyeDetector, upperRightCroppedFace);
                    end
                
                    if ~isempty(rightEyeRegion)
                        [~, prominentIdx] = max(calculateArea(rightEyeRegion));
                        rightEyeRect = rightEyeRegion(prominentIdx, :);
                
                        scaleFactor = 1.1;
                        newWidth = rightEyeRect(3) * scaleFactor;
                        newHeight = rightEyeRect(4) * scaleFactor;

                        deltaX = (newWidth - rightEyeRect(3)) / 2;
                        deltaY = (newHeight - rightEyeRect(4)) / 2;
                        newX = rightEyeRect(1) - deltaX;
                        newY = rightEyeRect(2) - deltaY;

                        rightEyeRect = [newX, newY, newWidth, newHeight];
                
                        rightEyeRect(1:2) = rightEyeRect(1:2) + [upperWidth/(2+(2/9)) 0] + upperHalfFace(1:2);
                        rightEyeRect = rightEyeRect/scale;
                        displayImage = insertShape(displayImage, "Rectangle", rightEyeRect, LineWidth=lineWidth, Color="Blue");
                    end

                    if fallback && isempty(leftEyeRegion) && isempty(rightEyeRegion)
                        % If all detection fails and the "Enable Fallback" checkbox is checked, use geometric analysis to highlight the features instead
                        eyeRect = [faceRect(1) + faceRect(3) * 0.25, faceRect(2) + faceRect(4) * 0.25, faceRect(3) * 0.5, faceRect(4) * 0.2];
                        eyeRect = eyeRect/scale;
                        displayImage = insertShape(displayImage, "Rectangle", eyeRect, LineWidth=lineWidth, Color="Blue");
                    end
                end
            end
            
            if mouth % Only detect the mouth if the "Highlight Mouth" checkbox is checked
                
                % Only look for the mouth in the lower half of the face, (helps with accuracy and performance)
                lowerHalfFace = faceRect;
                lowerHalfFace(2) = lowerHalfFace(2) + lowerHalfFace(4)/2;
                lowerHalfFace(4) = lowerHalfFace(4)/2;
                lowerCroppedFace = imcrop(processedImage, lowerHalfFace);
                mouthDetector = vision.CascadeObjectDetector("Mouth", MergeThreshold=1);
                mouthRegion = step(mouthDetector, lowerCroppedFace);

                if ~isempty(mouthRegion)
                    newMouthRegion = [];
                    for j = 1:size(mouthRegion,1)
                        % Checks if the ratio of the mouth is reasonable and that it isn't detected within the eyes region
                        if (~exist("eyeRect", "var") || ~checkRectOverlap(mouthRegion(j, :), eyeRect)) && calculateProportion(mouthRegion(j, :), faceArea) > 0.05
                            newMouthRegion = [newMouthRegion, mouthRegion(j, :)]; %#ok<AGROW>
                        end
                    end
                    if ~isempty(newMouthRegion)
                        mouthRegion = newMouthRegion;
                    
                    else
                        if fallback
                            % If all detection fails and the "Enable Fallback" checkbox is checked, use geometric analysis to highlight the features instead
                            mouthRect = [faceRect(1) + faceRect(3) * 0.25, faceRect(2) + faceRect(4) * 0.75, faceRect(3) * 0.5, faceRect(4) * 0.1];
                            mouthRegion = mouthRect;
                        end
                    end

                    % Picks the biggest mouth detected
                    [~, prominentIdx] = max(calculateArea(mouthRegion));
                    mouthRect = mouthRegion(prominentIdx, :);

                    % Makes the box larger by 10%, to be certain it covers the correct area
                    scaleFactor = 1.1;
                    newWidth = mouthRect(3) * scaleFactor;
                    newHeight = mouthRect(4) * scaleFactor;
                
                    deltaX = (newWidth - mouthRect(3)) / 2;
                    deltaY = (newHeight - mouthRect(4)) / 2;
                    newX = mouthRect(1) - deltaX;
                    newY = mouthRect(2) - deltaY;
                
                    mouthRect = [newX, newY, newWidth, newHeight];
                
                    mouthRect(1:2) = mouthRect(1:2) + lowerHalfFace(1:2);
                    mouthRect = mouthRect/scale;
                    
                    % Draw a box around the detected pair
                    displayImage = insertShape(displayImage, "Rectangle", mouthRect, LineWidth=lineWidth, Color="Green");
                
                elseif fallback
                    % If all detection fails and the "Enable Fallback" checkbox is checked, use geometric analysis to highlight the features instead
                    mouthRect = [faceRect(1) + faceRect(3) * 0.25, faceRect(2) + faceRect(4) * 0.75, faceRect(3) * 0.5, faceRect(4) * 0.1];
                    mouthRect = mouthRect/scale;
                    displayImage = insertShape(displayImage, "Rectangle", mouthRect, LineWidth=lineWidth, Color="Green");
                end
            end

            if nose % Only detect the nose if the "Highlight Nose" checkbox is checked
                middleHalfFace = faceRect;

                % Only look for the nose in the middle half of the face, (helps with accuracy and performance)
                middleHalfFace(2) = middleHalfFace(2) + middleHalfFace(4)/4;
                middleHalfFace(4) = middleHalfFace(4)/2;
                middleCroppedFace = imcrop(processedImage, middleHalfFace);
                noseDetector = vision.CascadeObjectDetector("Nose", MergeThreshold=1);
                noseRegion = step(noseDetector, middleCroppedFace);

                if ~isempty(noseRegion)
                    newNoseRegion = [];
                    % Checks if the ratio of the nose is reasonable and that it isn't within neither the eyes nor the mouth regions
                    for j = 1:size(noseRegion,1)
                        if (~exist("eyeRect", "var") || ~checkRectOverlap(noseRegion(j, :), eyeRect)) && (~exist("mouthRect", "var") || ~checkRectOverlap(noseRegion(j, :), mouthRect)) && calculateProportion(noseRegion(j, :), faceArea) > 0.05
                            newNoseRegion = [newNoseRegion, noseRegion(j, :)]; %#ok<AGROW>
                        end
                    end
                    if ~isempty(newNoseRegion)
                        noseRegion = newNoseRegion;
                    
                    else
                        if fallback
                            % If all detection fails and the "Enable Fallback" checkbox is checked, use geometric analysis to highlight the features instead
                            noseRect = [faceRect(1) + faceRect(3) * 0.3, faceRect(2) + faceRect(4) * 0.5, faceRect(3) * 0.4, faceRect(4) * 0.2];
                            noseRegion = noseRect;
                        end
                    end

                    % Picks the biggest nose detected
                    [~, prominentIdx] = max(calculateArea(noseRegion));
                    noseRect = noseRegion(prominentIdx, :);

                    % Makes the box larger by 10%, to be certain it covers the correct area
                    scaleFactor = 1.1;
                    newWidth = noseRect(3) * scaleFactor;
                    newHeight = noseRect(4) * scaleFactor;
                
                    deltaX = (newWidth - noseRect(3)) / 2;
                    deltaY = (newHeight - noseRect(4)) / 2;
                    newX = noseRect(1) - deltaX;
                    newY = noseRect(2) - deltaY;
                
                    noseRect = [newX, newY, newWidth, newHeight];

                    noseRect(1:2) = noseRect(1:2) + middleHalfFace(1:2);
                    noseRect = noseRect/scale;
                    
                    % Draw a box around the detected pair
                    displayImage = insertShape(displayImage, "Rectangle", noseRect, LineWidth=lineWidth, Color="Yellow");

                elseif fallback
                    % If all detection fails and the "Enable Fallback" checkbox is checked, use geometric analysis to highlight the features instead
                    noseRect = [faceRect(1) + faceRect(3) * 0.3, faceRect(2) + faceRect(4) * 0.5, faceRect(3) * 0.4, faceRect(4) * 0.2];
                    noseRect = noseRect/scale;
                    displayImage = insertShape(displayImage, "Rectangle", noseRect, LineWidth=lineWidth, Color="Yellow");
                end
            end
        end

        % Rotate the image back if it was rotated
        if angle ~= 0
            displayImage = imrotate(displayImage, (-angle));
        end


        % Display the results
        axes(oldImageFrame);
        imshow(displayProcessedImage);

        axes(newImageFrame);
        imshow(displayImage);

    end

    function runDetection(~, ~)
        % Update values from the GUI and run the detection

        defaultSize = ceil(slider1.Value);
        defaultGamma = slider2.Value;
        defaultScale = slider3.Value;
        defaultLighten = slider4.Value;
        defaultDarken = slider5.Value;
        faceThreshold = ceil(slider6.Value);
        fallback = fallbackCheckbox.Value;
        face = faceCheckbox.Value;
        eyes = eyesCheckbox.Value;
        nose = noseCheckbox.Value;
        mouth = mouthCheckbox.Value;
        rotate = rotateCheckbox.Value;

        faceDetect(imageFile, fallback, face, eyes, nose, mouth, rotate, defaultSize, defaultGamma, defaultScale, defaultLighten, defaultDarken, faceThreshold);
    end

    function selectImage(~, ~)
        % Launch the file chooser
        [filename, pathname] = uigetfile({"*.*", "All Files"; "*.jpg;*.png;*.bmp", "Images (*.jpg, *.png, *.bmp)"}, "Select an Image File");
        if ~isequal(filename, 0) || isequal(pathname, 0)
            imageFile = fullfile(pathname, filename);
            runDetection;
        end
    end

    function image = readImageAsRGB(imageFile)
        % Load all images as RGB
        [image, map] = imread(imageFile);
        if size(image, 3) == 1
            if isempty(map)
                image = cat(3, image, image, image);
            else
                image = ind2rgb(image, map);
            end
        end
        if islogical(image)
            rgbImage = zeros(size(image, 1), size(image, 2), 3, "uint8");
        
            rgbImage(:, :, :) = image * 255;
            image = rgbImage;
        end
    end
    
    function image = downScaleLargeImage(originalImage)
        % Downscale any image larger than 1920x1080
        image = originalImage;
    
        [height, width, ~] = size(originalImage);
            
        max_width = 1920;
        max_height = 1080;
    
        scale_width = max_width / width;
        scale_height = max_height / height;
    
        scale_factor = min(scale_width, scale_height);
    
        if scale_factor < 1
            image = imresize(originalImage, scale_factor);
        end
    end
    
    function processedImage = processImage(originalImage, gamma, scale, defaultLighten, defaultDarken)
        % Read the manual for more information on the preprocessing steps
        processedImage = rgb2gray(originalImage);
        processedImage = im2double(processedImage);
        processedImage = histeq(processedImage);
        processedImage = imsharpen(processedImage);
    
        smoothedSobel = imfilter(edge(processedImage, "sobel"), fspecial("average", [3 3]), "replicate");
        laplacian = processedImage + imfilter(processedImage, fspecial("laplacian", 1), "replicate");
        processedImage = processedImage + (laplacian .* smoothedSobel);
        
        processedImage = imadjust(processedImage, [], [], defaultLighten);
        processedImage = adapthisteq(processedImage);
        processedImage = imadjust(processedImage, [], [], defaultDarken);
        processedImage = adapthisteq(processedImage);
        processedImage = imgaussfilt(processedImage, gamma);
        processedImage = imresize(processedImage, scale);
    end
    
    function [faces, rotatedImage, angle] = detectTiltedFaces(processedImage, faceDetector)
        % This code is for the "Detect Titled Faces" checkbox
        % It basically rotates the image 22.5 degrees (both directions) unitl it finds a face(s)
        % And then the feature detection can be done on the detected faces of the rotated image
        angles = [22.5, -22.5, 45.0, -45.0, 67.5, -67.5, 90.0, -90.0, 112.5, -112.5, 135.0];
        for i = 1:length(angles)
            angle = angles(i);
            rotatedImage = imrotate(processedImage, angle);
            faces = step(faceDetector, rotatedImage);
            if ~isempty(faces)
                break;
            end
        end
    end
    
    function isWithin = checkRectOverlap(rect1, rect2)
        % Check if the first [x, y, height, width] object is within the second

        intersectX = max(rect1(1), rect2(1));
        intersectY = max(rect1(2), rect2(2));
        intersectW = min(rect1(1) + rect1(3), rect2(1) + rect2(3)) - intersectX;
        intersectH = min(rect1(2) + rect1(4), rect2(2) + rect2(4)) - intersectY;
    
        if intersectW > 0 && intersectH > 0
            intersectionArea = intersectW * intersectH;
            rect2Area = rect2(3) * rect2(4);
    
            coverage = (intersectionArea / rect2Area) * 100;
    
            isWithin = coverage >= 75;
        else
            isWithin = false;
        end
    end
end