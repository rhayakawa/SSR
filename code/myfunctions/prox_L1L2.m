function eta = prox_L1L2(u,gamma)
% prox_L1L2 proximity operator for the difference of L1 and L2 norms

  [infNorm,index_max]=max(abs(u));
  if infNorm>gamma
    z=sign(u).*max(abs(u)-gamma,0);
    norm_z=norm(z,2);
    eta=(norm_z+gamma)/norm_z.*z;
  elseif infNorm>0
    eta=zeros(size(u));
    eta(index_max)=u(index_max);
  else
    eta=zeros(size(u));
  end
  
end
