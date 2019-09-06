function eta = prox_L1_2(u,gamma)
%prox_L1_2 proximity operator of L1/2 norm

  eta=2/3*u.*(1+cos(2/3*acos(-3^(3/2)/4*gamma.*abs(u).^(-3/2))));
  eta(abs(u)<3/2*gamma.^(2/3))=0;

end

