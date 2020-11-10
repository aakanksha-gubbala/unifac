clear
clc
% (1) Water - (2) Acetic acid - (3) Butyl acetate
    
N = 100; % Number of data points
T = 298.16; % Temperature in K
    
nu = [0, 1, 1;  % CH3
      0, 0, 3;  % CH2
      0, 0, 1;  % CH3COO
      0, 1, 0;  % COOH
      1, 0, 0]; % H2O

R = [0.9011, 0.6744, 1.9031, 1.3013, 0.92];

Q = [0.848, 0.540, 1.728 , 1.224, 1.4];

     % CH3  CH2    CH3COO  COOH    H2O
a = [0    , 0    , 232.1 , 663.5 , 1318  ; % CH3
     0    , 0    , 232.1 , 663.5 , 1318  ; % CH2
     114.8, 114.8, 0     , 660.2 , 200.8 ; % CH3COO
     315.3, 315.3, -256.3, 0     , -66.17; % COOH
     300  , 300  , 72.87 , -14.09, 0    ]; % H2O

r = R * nu;
q = Q * nu;

% Ji = ri / sum(rj xj)
% Li = qi / sum(qj xj)
% theta_i = xi qi / sum(xj qj)
% ln(gamma_c) = 1 - Ji + ln(Ji) - 5qi(1 - Ji/Li + ln(Ji/Li))
% ln(gamma_r) = qi(1 - sum_k (theta_k * beta_ik/s_k - e_ki ln(beta_ik/s_k)))

x = zeros(N + 1, 3);
i = 1;

for x1 = 0:(1/N):1
    for x2 = 0:(1/N):(1-x1)
        x(i, :) = [x1, x2, 1 - x1- x2];
        i = i + 1;
    end
end

J = zeros(length(x), 3);

for i = 1:length(x)
    J(i, :) =  r ./ sum(x(i, :) .* r);
end

L = zeros(length(x), 3);

for i = 1:length(x)
    L(i, :) =  q ./ sum(x(i, :) .* q);
end

gamma_c =  exp(1 - J + log(J) - 5 .* q .* (1 - J./L + log(J ./ L)));

e = zeros(5, 3);

for i = 1:3
    e(:, i) = nu(:, i) .* Q' ./ q(i);
end

tau = exp(-a ./ T);

beta = e' * tau;

theta = zeros(length(x) , 5);

for i = 1:length(x)
   for j = 1:5
       theta(i, j) = sum(x(i, :) .* q .* e(j, :)) ./ sum(x(i, :) .* q);     
   end
end

s = theta * tau;

gamma_r = zeros(length(x), 3);

for i = 1:length(x)
    gamma_r(i, :) = exp(q .* (1 - (sum((theta(i, :) .* beta ./ s(i, :))' - log(beta ./ s(i, :))' .* e))));
end

gamma = gamma_c .* gamma_r;
