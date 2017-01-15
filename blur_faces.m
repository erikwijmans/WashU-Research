function blur_faces()
    fileID = fopen('/Users/erikwijmans/Projects/3DScanData/DUC/Floor1/binaryFiles/DUC_binary_000.dat', 'r');
    tmp = fread(fileID, 2, 'uint32');
    fclose(fileID);
    cols = tmp(1);
    rows = tmp(2);
    


    mm = memmapfile('/Users/erikwijmans/Projects/3DScanData/DUC/Floor1/binaryFiles/DUC_binary_000.dat',...
        'Offset', 8,...
        'Format', {'single',[3 1],'pos';...
        'single',[1 1],'I';...
        'uint8',[3 1],'color'}, 'Repeat', rows*cols);

    pointcloud = mm.Data;

    disp([rows cols])
    C = zeros(cols, rows, 3, 'uint8');

    row = rows;
    col = round(0.995*cols/2.0);

    for k=1:rows*cols
        C(col, row, :) = pointcloud(k).color;
        if row == 1
            row = rows - 1;
            if col == 1
                col = cols;
            else
                col = col - 1;
            end
        else
            row = row - 1;
        end
    end

    f = figure;
    image(C)
    hold on

    rects = double.empty(0, 4);
    function selectrect()
        r = round(getrect(f));
        width = r(3);
        height = r(4);
        for j = 0:height - 1
            for i = 0:width - 1
                C(r(2) + j, r(1) + i, 1) = round(C(r(2) + j, r(1) + i, 1) .* 0.5 + 0.5 .* 128);
                C(r(2) + j, r(1) + i, 2) = round(C(r(2) + j, r(1) + i, 2) .* 0.5);
                C(r(2) + j, r(1) + i, 3) = round(C(r(2) + j, r(1) + i, 3) .* 0.5);
            end
        end
        rects(end + 1, :) = r;
        image(C);
    end
%     selectrect()
end
