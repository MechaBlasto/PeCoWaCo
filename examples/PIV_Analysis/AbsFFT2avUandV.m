function M = AbsFFT2avUandV(array1, array2)

M = zeros([60 1]);
totalcoordinate = size(array1{1,1});
 
startcoordinatek = 1; % to remove nonoscilatory regions
startcoordinatel = 1; % to remove nonoscilatory regions

endcoordinatek = totalcoordinate(1,1) ; % to remove nonoscilatory regions
endcoordinatel = totalcoordinate(1,2) ; % to remove nonoscilatory regions
counter = 0;
for k = startcoordinatek:endcoordinatek
        for l = startcoordinatel:endcoordinatel
         single = ExtractPIV(k,l,array1);
            if any(isnan(single))
             continue
            else
             M = M + power(abs(fft(single)),2);
             counter = counter + 1;
            end
         end        
end

N = zeros([60 1]);
totalcoordinate = size(array2{1,1});
 
startcoordinatek = 1; % to remove nonoscilatory regions
startcoordinatel = 1; % to remove nonoscilatory regions

endcoordinatek = totalcoordinate(1,1) ; % to remove nonoscilatory regions
endcoordinatel = totalcoordinate(1,2) ; % to remove nonoscilatory regions
counter2 = 0 ;
for k = startcoordinatek:endcoordinatek
        for l = startcoordinatel:endcoordinatel
         single = ExtractPIV(k,l,array2);
            if any(isnan(single))
             continue
            else
             N = N + power(abs(fft(single)),2);
             counter2 = counter2 + 1;
            end
         end        
end


M = sqrt(M)/counter;
N = sqrt(N)/counter2;
Fs = 0.1; % Frequency of imaging
T = 1/Fs; % Period of imaging
L = 60;  % Number of points in signal
t = (0:L-1)*T;



P1 = M(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);

P2 = N(1:L/2+1);
P2(2:end-1) = 2*P2(2:end-1);

P3 = (P1 + P2 )/2 
f = Fs*(0:(L/2))/L

plot(f,P3) 

title('FFT U and V component together')
xlabel('f (Hz)')
ylabel('Amplitude (um/s)')
 

 
