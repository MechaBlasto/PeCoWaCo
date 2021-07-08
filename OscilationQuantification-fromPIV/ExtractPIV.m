function A = ExtractPIV(r,c,array)

nRows = size(array,1);
A = NaN(nRows,1);


for i = 1:nRows
    arrayAtI = array{i,1};
    
    if r > size(arrayAtI,1) || c > size(arrayAtI,2)
        disp('ERROR: r or c too big');
        return;
    else
        elAtI = arrayAtI(r,c);
    end
    
    A(i,1) = elAtI;
    
end