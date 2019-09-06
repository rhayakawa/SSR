function eta = prox_L0(u,gamma)
%prox_L0 proximity operator of L0 norm

  eta=u;
  eta(abs(u)<sqrt(2*gamma))=0;

end

