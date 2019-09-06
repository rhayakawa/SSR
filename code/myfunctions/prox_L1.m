function eta = prox_L1(u,gamma)
%prox_L1 proximity operator of L1 norm

  eta=sign(u).*max(abs(u)-gamma,0);

end
