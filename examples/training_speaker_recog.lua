require 'paths'
require 'rnn'
require 'torchx'
require 'mattorch'
require 'cunn'
require 'cutorch'

local dl = require 'dataload'

version = 2

--[[ command line arguments ]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a Language Model on PennTreeBank dataset using RNN or LSTM or GRU')
cmd:text('Example:')
cmd:text('th recurrent-language-model.lua --cuda --device 2 --progress --cutoff 4 --seqlen 10')
cmd:text("th recurrent-language-model.lua --progress --cuda --lstm --seqlen 20 --hiddensize '{200,200}' --batchsize 20 --startlr 1 --cutoff 5 --maxepoch 13 --schedule '{[5]=0.5,[6]=0.25,[7]=0.125,[8]=0.0625,[9]=0.03125,[10]=0.015625,[11]=0.0078125,[12]=0.00390625}'")
cmd:text("th recurrent-language-model.lua --progress --cuda --lstm --seqlen 35 --uniform 0.04 --hiddensize '{1500,1500}' --batchsize 20 --startlr 1 --cutoff 10 --maxepoch 50 --schedule '{[15]=0.87,[16]=0.76,[17]=0.66,[18]=0.54,[19]=0.43,[20]=0.32,[21]=0.21,[22]=0.10}' -dropout 0.65")
cmd:text('Options:')
-- training
cmd:option('--startlr', 0.005, 'learning rate at t=0')
cmd:option('--minlr', 0.000001, 'minimum learning rate')
cmd:option('--saturate', 250, 'epoch at which linear decayed LR will reach minlr')
cmd:option('--schedule', '', 'learning rate schedule. e.g. {[5] = 0.004, [6] = 0.001}')
cmd:option('--momentum', 0.9, 'momentum')
cmd:option('--maxnormout', -1, 'max l2-norm of each layer\'s output neuron weights')
cmd:option('--cutoff', -1, 'max l2-norm of concatenation of all gradParam tensors')
cmd:option('--batchSize', 12, 'number of examples per batch')
cmd:option('--cuda', true, 'use CUDA')
cmd:option('--device', 1, 'sets the device (GPU) to use')
cmd:option('--maxepoch', 500, 'maximum number of epochs to run')
cmd:option('--earlystop', 50, 'maximum number of epochs to wait to find a better local minima for early-stopping')
cmd:option('--progress', false, 'print progress bar')
cmd:option('--silent', false, 'don\'t print anything to stdout')
cmd:option('--uniform', 0.1, 'initialize parameters using uniform distribution between -uniform and uniform. -1 means default initialization')
-- rnn layer 
cmd:option('--lstm', true, 'use Long Short Term Memory (nn.LSTM instead of nn.Recurrent)')
cmd:option('--gru', false, 'use Gated Recurrent Units (nn.GRU instead of nn.Recurrent)')
cmd:option('--seqlen', 5, 'sequence length : back-propagate through time (BPTT) for this many time-steps')
cmd:option('--hiddensize', '{25,25}', 'number of hidden units used at output of each recurrent layer. When more than one is specified, RNN/LSTMs/GRUs are stacked')
cmd:option('--dropout', 0, 'apply dropout with this probability after each rnn layer. dropout <= 0 disables it.')
-- data
cmd:option('--batchsize', 12, 'number of examples per batch')
cmd:option('--trainsize', -1, 'number of train examples seen between each epoch')
cmd:option('--validsize', -1, 'number of valid examples used for early stopping and cross-validation') 
cmd:option('--savepath', paths.concat(dl.SAVE_PATH, 'rnnlm'), 'path to directory where experiment log (includes model) will be saved')
cmd:option('--id', '', 'id string of this experiment (used to name output file) (defaults to a unique id)')

cmd:text()
local opt = cmd:parse(arg or {})
opt.hiddensize = loadstring(" return "..opt.hiddensize)()
opt.schedule = loadstring(" return "..opt.schedule)()
if not opt.silent then
   table.print(opt)
end
opt.id = opt.id == '' and ('ptb' .. ':' .. dl.uniqueid()) or opt.id

local lm= nn.Sequential()
local inputsize = 39--opt.hiddensize[1]

for i,hiddensize in ipairs(opt.hiddensize) do 
   --local rnn
   require 'nngraph'
   --nn.FastLSTM.usenngraph = true -- faster
   lm:add(nn.FastLSTM(inputsize, hiddensize,3))
   inputsize = hiddensize
end

batchSize = 12
inputsize=39
rho_v = 6000

files = paths.indexdir('data', 'mat', true)
--lm:add(nn.Sequencer(stepmodule))
lm:add(nn.Linear(25,12))
lm:add(nn.LogSoftMax())

lm= nn.Sequencer(lm)
local finalsize= opt.hiddensize[#opt.hiddensize]
local rho= 20
nIndex=10;

print"Language Model:"
print(lm)

if opt.uniform > 0 then
   for k,param in ipairs(lm:parameters()) do
      param:uniform(-opt.uniform, opt.uniform)
   end
end
local targetmodule = nn.SplitTable(1)
--if opt.cuda then
--   targetmodule = nn.Sequential()
--      :add(nn.Convert())
--      :add(targetmodule)
--end
 
--[[ CUDA ]]--
criterion= nn.SequencerCriterion(nn.ClassNLLCriterion())
--criterion = nn.ClassNLLCriterion()
if opt.cuda then
  require 'cunn'
   cutorch.setDevice(opt.device)
   print("why cuda")
   lm:cuda()
   criterion:cuda()
   targetmodule:cuda()
   
end

--[[ experiment log ]]--

-- is saved to file every time a new validation minima is found
local xplog = {}
xplog.opt = opt -- save all hyper-parameters and such
xplog.dataset = 'random'
xplog.vocab = {}--trainset.vocab
-- will only serialize params
xplog.model = nn.Serial(lm)
xplog.model:mediumSerial()
--xplog.model = lm
xplog.criterion = criterion
xplog.targetmodule = targetmodule
-- keep a log of NLL for each epoch
xplog.trainppl = {}
xplog.valppl = {}
-- will be used for early-stopping
xplog.minvalppl = 99999999
xplog.epoch = 0
local ntrial = 0
paths.mkdir(opt.savepath)

inputs_v={}
targets_v={}


temptarget = torch.zeros(batchSize)

temptarget[1]=6
temptarget[2]=12
temptarget[3]=10
temptarget[4]=11
temptarget[5]=1
temptarget[6]=4
temptarget[7]=2
temptarget[8]=3
temptarget[9]=5
temptarget[10]=9
temptarget[11]=8
temptarget[12]=7
	
local epoch = 1
opt.lr = opt.startlr
--opt.trainsize = opt.trainsize == -1 and trainset:size() or opt.trainsize
--opt.validsize = opt.validsize == -1 and validset:size() or opt.validsize

while opt.maxepoch <= 0 or epoch <= opt.maxepoch do
   print("")
   print("Epoch #"..epoch.." :")

   -- 1. training
   
   local a = torch.Timer()
   lm:training()
   local sumErr = 0
   --for i, inputs, targets in trainset:subiter(opt.seqlen, opt.trainsize) do
   x = 1
   for i=1,1500 do
   print("Iteration #"..i.." :")
      --local inputs= torch.rand(rho,opt.batchsize,inputsize)
      --local inputs= torch.rand(2,39)
      --local inputs2= torch.rand(4,2,39)

      -- local targets= torch.zeros(rho,batchSize,1)
      -- local inputs = torch.zeros(rho,batchSize,inputsize)
      -- targets_v= torch.zeros(rho_v,batchSize,1)
      -- inputs_v= torch.zeros(rho_v,batchSize,inputsize)

      local inputs={}
      local targets={}   
    
  

      for m= 1,rho do 
      	inputs[m]= torch.zeros(batchSize,inputsize)
      	targets[m] = torch.zeros(batchSize)
      end
      
      for m= 1, rho_v do
      	inputs_v[m]= torch.zeros(batchSize,inputsize)
      	targets_v[m] = torch.zeros(batchSize)
      end
      
      for q = 1,batchSize do
	    -- print(files:filename(q))
	    tmp = mattorch.load(files:filename(q)).final_ft:transpose(1,2)
	    --print("-------------------------------Mind")
	    --print(tmp:size())
	    hehe = 1
	    for p = (((i-1)*rho)+1),i*rho do
	    -- for p = 1, rho do
        	inputs[hehe][q] = tmp[p]        	
        	hehe = hehe+1
	    end
	    for p = 1,rho_v do
        	inputs_v[p][q] = tmp[30000+p] 
	    end 
      end
     
      for p = 1,rho do
	    inputs[p]=inputs[p]:cuda() 
      end
      
      for p = 1,rho do
	    targets[p] = temptarget
      end
      
      for p = 1,rho do
	    targets[p]=targets[p]:cuda() 
      end
      
      
      for p = 1, rho_v do
            targets_v[p] = temptarget
      end
      
     
      --[[for i=1,3 do -- time step
        targets[i]= torch.Tensor(2):fill(math.ceil(math.random()*10))
        inputs[i]= torch.rand(2,39)
      end--]]
      --print(inputs)
--      targets = targetmodule:forward(targets)
      --print(targets)     
      -- forward
      --local outputs_1 = lm:forward(inputs2) -- for checking
      --print("starting lmmodule *****************************************************")
      local outputs = lm:forward(inputs)
      --print(outputs)
      --print("past lmmodule *****************************************************")
      local err = criterion:forward(outputs, targets)
      sumErr = sumErr + err
      
       local ppl = torch.exp(sumErr/rho*batchSize)
       -- print("Training PPL : "..ppl)
       print("Training Error : "..torch.exp(err/rho*batchSize)) 	
       
      --print("past criterion *****************************************************")
      -- backward 
      local gradOutputs = criterion:backward(outputs, targets)
      -- print("past backcriterion *****************************************************")
      lm:zeroGradParameters()
      lm:backward(inputs, gradOutputs)
      
      -- print("past back *****************************************************")
      -- update
      if opt.cutoff > 0 then
         local norm = lm:gradParamClip(opt.cutoff) -- affects gradParams
         opt.meanNorm = opt.meanNorm and (opt.meanNorm*0.9 + norm*0.1) or norm
      end
      lm:updateGradParameters(opt.momentum) -- affects gradParams
      lm:updateParameters(opt.lr) -- affects params
      lm:maxParamNorm(opt.maxnormout) -- affects params

      if opt.progress then
         xlua.progress(math.min(i + opt.seqlen, opt.trainsize), opt.trainsize)
      end

      if i % 1000 == 0 then
         collectgarbage()
      end

   end
   
   -- learning rate decay
   if opt.schedule then
      opt.lr = opt.schedule[epoch] or opt.lr
   else
      opt.lr = opt.lr + (opt.minlr - opt.startlr)/opt.saturate
   end
   opt.lr = math.max(opt.minlr, opt.lr)
   
   if not opt.silent then
      print("learning rate", opt.lr)
      if opt.meanNorm then
         print("mean gradParam norm", opt.meanNorm)
      end
   end

   if cutorch then cutorch.synchronize() end
   local speed = a:time().real
   print(string.format("Speed : %f sec/batch ", speed))

   xplog.trainppl[epoch] = ppl
   

   -- 2. cross-validation

   lm:evaluate()
   --local sumErr = 0
   --[[
   for i, inputs, targets in validset:subiter(opt.seqlen, opt.validsize) do
      targets = targetmodule:forward(targets)
      local outputs = lm:forward(inputs)
      local err = criterion:forward(outputs, targets)
      sumErr = sumErr + err
   end
   --]]
   
   for p = 1,rho_v do
     inputs_v[p]=inputs_v[p]:cuda() 
   end     
    
  for p = 1,rho_v do
    targets_v[p]=targets_v[p]:cuda() 
  end
      
      
   --print(targets_v)
   -- targets_v = targetmodule:forward(targets_v)
   local outputs_v = lm:forward(inputs_v)
   local err = criterion:forward(outputs_v, targets_v)   
   local ppl = torch.exp(err/rho_v)
   --local ppl = torch.exp(sumErr/opt.validsize)
   print("Validation PPL : "..ppl)

   xplog.valppl[epoch]= ppl
   ntrial = ntrial + 1

   if epoch%5 ==0 then
   	xplog.epoch = epoch 
   	xplog.minvalppl = ppl
   	print("Evaluate model using : ")
print("th scripts/evaluate-rnnlm.lua --xplogpath "..paths.concat(opt.savepath, opt.id..'.t7')..(opt.cuda and '--cuda' or ''))
   	local filename = paths.concat(epoch, opt.savepath, opt.id..'.t7')
   	torch.save(filename, xplog)
   end

   collectgarbage()
   epoch = epoch + 1
end

