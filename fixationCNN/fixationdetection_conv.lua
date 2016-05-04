require 'dp'
require 'torchx'

--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Facial Keypoint detector using Convolution Neural Network Training/Optimization')
cmd:text('Example:')
cmd:text('$> th facialkeypointdetector.lua --batchSize 128 --momentum 0.5')
cmd:text('Options:')
cmd:option('--learningRate', 0.9, 'learning rate at t=0')
cmd:option('--maxOutNorm', 2, 'max norm each layers output neuron weights')
cmd:option('--momentum', .9, 'momentum')
cmd:option('--padding', true, 'add math.floor(kernelSize/2) padding to the input of each convolution') 
cmd:option('--channelSize', '{32,64,96,128}', 'Number of output channels for each convolution layer.')
cmd:option('--kernelSize', '{5,5,5,5,5}', 'kernel size of each convolution layer. Height = Width')
cmd:option('--kernelStride', '{1,1,1,1,1}', 'kernel stride of each convolution layer. Height = Width')
cmd:option('--poolSize', '{2,2,2,2,2}', 'size of the max pooling of each convolution layer. Height = Width')
cmd:option('--poolStride', '{2,2,2,2,2}', 'stride of the max pooling of each convolution layer. Height = Width')
cmd:option('--hiddenSize', '{3000, 1000}', 'size of the dense hidden layers (after convolutions, before output)')
cmd:option('--batchSize', 128, 'number of examples per batch')
cmd:option('--cuda', true, 'use CUDA')
cmd:option('--useDevice', 1, 'sets the device (GPU) to use')
cmd:option('--maxEpoch', 200, 'maximum number of epochs to run')
cmd:option('--maxTries', 30, 'maximum number of epochs to try to find a better local minima for early-stopping')
cmd:option('--xpPath', '', 'path to a previously saved model')
cmd:option('--dropout', true, 'apply dropout on hidden neurons')
cmd:option('--batchNorm', false, 'use batch normalization. dropout is mostly redundant with this')
cmd:option('--standardize', true, 'apply Standardize preprocessing')
cmd:option('--zca', false, 'apply Zero-Component Analysis whitening')
cmd:option('--activation', 'ReLU', 'transfer function like ReLU, Tanh, Sigmoid')
cmd:option('--dropout', true, 'use dropout')
cmd:option('--dropoutProb', '{0.2,0.2,0.5,0.5,0.8,.8}', 'dropout probabilities')
cmd:option('--accUpdate', false, 'accumulate gradients inplace')
cmd:option('--progress', true, 'print progress bar')
cmd:option('--validRatio', 1/10, 'proportion of dataset used for cross-validation')
cmd:option('--silent', false, 'dont print anything to stdout')
cmd:text()
opt = cmd:parse(arg or {})
if not opt.silent then
   table.print(opt)
end

--if opt.xpPath ~= '' then
 --  -- check that saved model exists
  -- assert(paths.filep(opt.xpPath), opt.xpPath..' does not exist')
--end

torch.setnumthreads(8)
torch.setdefaulttensortype('torch.FloatTensor')

--[[loading fixations data]]--
function FixationDetection(dataPath, validRatio)
   validRatio = validRatio or 0.15

   -- 1. load images into input and target Tensors
   local eye = paths.indexdir(paths.concat(dataPath))
   local size = eye:size()
   local shuffle = torch.randperm(size) -- 
   local input = torch.FloatTensor(size, 3, 96, 96)
   local target = torch.load(paths.concat(dataPath, 'labels.dat'))
   target = target[{{1,eye:size()},{}}] -- since not all trials were necessarily completed
   target = torch.FloatTensor(target)

   for i=1,eye:size() do
      local img = image.load(eye:filename(i))
         img = image.scale(img, 96, 96 )
         img = image.rgb2yuv(img)
      local idx = shuffle[i]
      input[idx]:copy(img)
      collectgarbage()
   end



   -- 2. divide into train and valid set and wrap into dp.Views

   nValid = math.floor(eye:size()*validRatio)
   nTrain = eye:size() - nValid
   
   -- construct trainer, 
   --y = target:narrow(1,1,nTrain)
   --y = target:clone()
   y = target
   Y = torch.FloatTensor(y:size(1), y:size(2), 98):zero()
   local pixels = torch.range(0,97)
   local stdv = 10
   local k = 0
   for i=1,y:size(1) do  -- for each pixel
      local keypoints = y[i]
      local new_keypoints = Y[i]
      for j=1,y:size(2) do  -- for each keypoint
         local kp = keypoints[j]
         if kp ~= -1 then
           if kp == 0 then
             kp = 1 
          end
          if kp >= 98 then
            kp = 98
          end
           local new_kp = new_keypoints[j]
            new_kp:add(pixels, -kp)
            new_kp:cmul(new_kp)
            new_kp:div(2*stdv*stdv)
            new_kp:mul(-1)
            new_kp:exp(new_kp)
            new_kp:div(math.sqrt(2*math.pi)*stdv)
        else
            k = k + 1
         end
      end
   end
   
   -- construct trainers
   local input_v, target_v = dp.ImageView(), dp.SequenceView()
   input_v:forward('bchw', input:narrow(1, 1, nTrain))
   target_v:forward('bwc', Y:narrow(1, 1, nTrain))
      
      -- construct valids
   local input_tar, target_tar = dp.ImageView(), dp.SequenceView()
   input_tar:forward('bchw', input:narrow(1, nTrain+1, nValid))
   target_tar:forward('bwc', Y:narrow(1, nTrain+1, nValid))
   
   -- 3. wrap views into datasets

   local train = dp.DataSet{inputs=input_v,targets=target_v,which_set='train'}
   local valid = dp.DataSet{inputs=input_tar,targets=target_tar,which_set='valid'}

   -- 4. wrap datasets into datasource

   local ds = dp.DataSource{train_set=train,valid_set=valid}
   ds:ioShapes('bchw', 'bwc')
   return ds
end


opt.channelSize = table.fromString(opt.channelSize)
opt.kernelSize = table.fromString(opt.kernelSize)
opt.kernelStride = table.fromString(opt.kernelStride)
opt.poolSize = table.fromString(opt.poolSize)
opt.poolStride = table.fromString(opt.poolStride)
opt.dropoutProb = table.fromString(opt.dropoutProb)
opt.hiddenSize = table.fromString(opt.hiddenSize)

--[[preprocessing]]--
local input_preprocess = {}
if opt.standardize then
   table.insert(input_preprocess, dp.Standardize())
end
if opt.zca then
   table.insert(input_preprocess, dp.ZCA())
end

--[[data]]--
ds = FixationDetection('data',0.15)

--[[Saved experiment]]--
if opt.xpPath ~= '' then
   if opt.cuda then
      require 'optim'
      require 'cunn'
      cutorch.setDevice(opt.useDevice)
   end
   xp = torch.load(opt.xpPath)
   if opt.cuda then
      xp:cuda()
   else
      xp:float()
   end
   print"running"
   xp:maxEpoch(xp:maxEpoch()+opt.maxEpoch)
   xp:run(ds)
   torch.save('experiment.dat', xp)
   os.exit()
end



--[[Model]]--
cnn = nn.Sequential()

-- convolutional and pooling layers
inputSize = ds:imageSize('c')
depth = 1
for i=1,#opt.channelSize do
   if opt.dropout and (opt.dropoutProb[depth] or 0) > 0 then
      -- dropout can be useful for regularization
      cnn:add(nn.SpatialDropout(opt.dropoutProb[depth]))
   end
   cnn:add(nn.SpatialConvolution(
      inputSize, opt.channelSize[i], 
      opt.kernelSize[i], opt.kernelSize[i], 
      opt.kernelStride[i], opt.kernelStride[i],
      opt.padding and math.floor(opt.kernelSize[i]/2) or 0
   ))
   if opt.batchNorm then
      -- batch normalization can be awesome
      cnn:add(nn.SpatialBatchNormalization(opt.channelSize[i]))
   end
   cnn:add(nn[opt.activation]())
   if opt.poolSize[i] and opt.poolSize[i] > 0 then
      cnn:add(nn.SpatialMaxPooling(
         opt.poolSize[i], opt.poolSize[i], 
         opt.poolStride[i] or opt.poolSize[i], 
         opt.poolStride[i] or opt.poolSize[i]
      ))
   end
   inputSize = opt.channelSize[i]
   depth = depth + 1
end
-- get output size of convolutional layers
outsize = cnn:outside{1,ds:imageSize('c'),ds:imageSize('h'),ds:imageSize('w')}
inputSize = outsize[2]*outsize[3]*outsize[4]
dp.vprint(not opt.silent, "input to dense layers has: "..inputSize.." neurons")

cnn:insert(nn.Convert(ds:ioShapes(), 'bchw'), 1)

-- dense hidden layers
cnn:add(nn.Collapse(3))
for i,hiddenSize in ipairs(opt.hiddenSize) do
   if opt.dropout and (opt.dropoutProb[depth] or 0) > 0 then
      cnn:add(nn.Dropout(opt.dropoutProb[depth]))
   end
   cnn:add(nn.Linear(inputSize, hiddenSize))
   if opt.batchNorm then
      cnn:add(nn.BatchNormalization(hiddenSize))
   end
   cnn:add(nn[opt.activation]())
   inputSize = hiddenSize
   depth = depth + 1
end

-- output layer
if opt.dropout and (opt.dropoutProb[depth] or 0) > 0 then
   cnn:add(nn.Dropout(opt.dropoutProb[depth]))
end
cnn:add(nn.Linear(inputSize, 2*98))
-- we use nn.MultiSoftMax() Module for detecting coordinates :
cnn:add(nn.Reshape(2,98))
cnn:add(nn.MultiSoftMax())

--[[Propagators]]--

logModule = nn.Sequential()
logModule:add(nn.AddConstant(0.00000001)) -- fixes log(0)=NaN errors
logModule:add(nn.Log())

train = dp.Optimizer{
   acc_update = opt.accUpdate,
   loss = nn.ModuleCriterion(nn.DistKLDivCriterion(), logModule, nn.Convert()),
   callback = function(model, report) 
      -- the ordering here is important
      if opt.accUpdate then
         model:accUpdateGradParameters(model.dpnn_input, model.output, opt.learningRate)
      else
         model:updateGradParameters(opt.momentum) -- affects gradParams
         model:updateParameters(opt.learningRate) -- affects params
      end
      model:maxParamNorm(opt.maxOutNorm) -- affects params
      model:zeroGradParameters() -- affects gradParams 
   end,
   sampler = dp.ShuffleSampler{batch_size = opt.batchSize},
   feedback = dp.FacialKeypointFeedback{precision=98},
   progress = opt.progress
}
valid = dp.Evaluator{
   sampler = dp.Sampler{batch_size = opt.batchSize},
   feedback = dp.FacialKeypointFeedback{precision=98}
}

--[[Experiment]]--
xp = dp.Experiment{
   model = cnn,
   optimizer = train,
   validator = valid,
   observer = {
      dp.FileLogger(),
      dp.EarlyStopper{
         error_report = {'validator','feedback','facialkeypoint','mse'},
         max_epochs = opt.maxTries
      }
   },
   random_seed = os.time(),
   max_epoch = opt.maxEpoch
}

--[[GPU or CPU]]--
if opt.cuda then
   require 'cutorch'
   require 'cunn'
   require 'cunnx'
   print('using CUDA')
   cutorch.setDevice(opt.useDevice)
   xp:cuda()
end

xp:verbose(not opt.silent)
if not opt.silent then
   print"Models :"
   print(cnn)
end


xp:run(ds)
torch.save('experiment.dat', xp)