function x_quantized = quantize(x,arrR)
  x_quantized=arrR(1)*ones(size(x));
  for index=2:length(arrR)
    x_quantized(x>(arrR(index-1)+arrR(index))/2)=arrR(index);
  end
end