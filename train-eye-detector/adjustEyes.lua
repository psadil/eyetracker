require 'torch'
require 'image'


local imagesAll = torch.Tensor(11382,3,120,80)

-- load eyes:
for f=1,11382 do
  imagesAll[f+1] = image.load('face-dataset/eyes/'..f..'.png') 
end

