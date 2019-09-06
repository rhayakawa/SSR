function eta = prox_L2_3(u,gamma)
%prox_L2_3 proximity operator of L2/3 norm

  z=(1/16*u.^(2)+sqrt(u.^(4)/256-8*gamma.^(3)/729)).^(1/3)+(1/16*u.^(2)-sqrt(u.^(4)/256-8*gamma.^(3)/729)).^(1/3);
  eta=sign(u)/8.*(sqrt(2*z)+sqrt(2*abs(u)./sqrt(2*z)-2*z)).^(3);
  eta(abs(u)<2*(2/3*gamma).^(3/4))=0;

end
