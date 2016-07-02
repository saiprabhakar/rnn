require 'paths'
require 'rnn'
require 'torchx'
require 'mattorch'
require 'cunn'
require 'cutorch'
require 'optim'
local dl = require 'dataload'

version = 2

--[[ command line arguments ]]--
cmd = torch.CmdLine()
cmd:option('--cuda', true, 'use CUDA')
cmd:option('--device', 1, 'sets the device (GPU) to use')
cmd:text()
local opt = cmd:parse(arg or {})
if not opt.silent then
   table.print(opt)
end
opt.id = opt.id == '' and ('ptb' .. ':' .. dl.uniqueid()) or opt.id

inputsize = 39--opt.hiddensize[1]
batchSize = 12
inputsize=39
rho_t = 6000
files = paths.indexdir('data', 'mat', true)
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

criterion= nn.SequencerCriterion(nn.ClassNLLCriterion())

if opt.cuda then
  require 'cunn'
   cutorch.setDevice(opt.device)
   print("why cuda")
   criterion:cuda()
end

inputs_t={}
targets_t={}  
      
      for m= 1, rho_t do
	inputs_t[m]= torch.zeros(batchSize,inputsize)
	targets_t[m] = torch.zeros(batchSize)
      end
      
      for q = 1,batchSize do
	    print(files:filename(q))
	    tmp = mattorch.load(files:filename(q)).final_ft:transpose(1,2)
	    for p = 1,rho_t do
        	inputs_t[p][q] = tmp[40000+p] 
	    end 
      end
     
      for p = 1,rho_t do
	    inputs_t[p]=inputs_t[p]:cuda() 
      end
      
      for p = 1,rho_t do
	    targets_t[p] = temptarget
      end
      
      for p = 1,rho_t do
	    targets_t[p]=targets_t[p]:cuda() 
      end      
           
subset = torch.load('/home/skonam/ML_Proj_deps/rnn/test/epoch_25.t7')
--print(subset.model)
lmodel = subset.model:cuda()
lmodel_o = lmodel.modules[1].modules[1].modules[1]
lmodel_o:remove()
lmodel_o:add(nn.SoftMax())
--print(lmodel_o)
--print(lmodel)
lmodel:cuda()

outputs_t = lmodel:forward(inputs_t)
outputs_tm={}  

for m= 1, rho_t do
	outputs_tm[m] = torch.zeros(batchSize)
end
      
for p = 1,rho_t do     
    sorted, indices = torch.sort(outputs_t[p],2 )
    outputs_tm[p] = indices:transpose(1,2)[12]
end 	

sumerr = 0

for p = 1,rho_t do     
    diff = outputs_tm[p]-targets_t[p]
    err = diff:eq(0):sum()/12
    sumerr = sumerr+err
end 
toterr = sumerr/rho_t
print(toterr) 
--	       1    2   3    4    5   6   7   8   9   10 11  12	
-- classes = {'6','12','10','11','1','4','2','3','5','9','8','7'}
-- confusion = optim.ConfusionMatrix(classes)
confusion = torch.zeros(12,12)
confusion_c = torch.zeros(12,12)	
	for i = 1,rho_t do
		for j=1,12 do
			confusion[j][outputs_tm[i][j]]= confusion[j][outputs_tm[i][j]]+1		           
            	end
        end
confusion_c[1] = confusion[5]
confusion_c[2] = confusion[7]
confusion_c[3] = confusion[8]
confusion_c[4] = confusion[6]
confusion_c[5] = confusion[9]
confusion_c[6] = confusion[1]
confusion_c[7] = confusion[12]
confusion_c[8] = confusion[11]
confusion_c[9] = confusion[10]
confusion_c[10] = confusion[3]
confusion_c[11] = confusion[4]
confusion_c[12] = confusion[2]
        
-- print confusion matrix
   print("Confusion")
   print(confusion_c)
   --trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
   --confusion:zero()
