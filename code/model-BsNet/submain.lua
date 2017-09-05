-- train subspectral, save pretraining states.

require './utils/init.lua'
ProFi = require 'utils/ProFi'
-- ================ Cmd Line  =======================
cmd = torch.CmdLine()
cmd:text()
cmd:text('Options')
-- data
cmd:option('-data', 	1, 			'1: Pines, 2: PaviaU; 3: Salinas')
cmd:option('-pixel', 	0,			'data: 0 pixel, other patch3')
cmd:option('-batchsize', 50,     	'batchsize')
cmd:option('-norm',  	0, 			'Normalization methods, 0: no norm; 1 dataset; 2 instance; 3 channel')

-- model
cmd:option('-resetw',	0,			'Reset init weight methods, range [0, 4]')
cmd:option('-full',  	0, 			'Run FUll model BP. ')
cmd:option('-dp',		0,		'Dropout percentage. 0 for no Dropout layer.')

-- training
cmd:option('-cuda', 	1, 			'0: disable, 1: enable')
cmd:option('-maxiter', 	2, 		'Max epoch of the BP part')
cmd:option('-optmsd',	2,			'Optimization methods, 0 SGD, 1 adagrad, other adadelta')
cmd:option('-lr', 		1e-2, 		'learning rate')
cmd:option('-lrDecay', 	1e-4, 		'learning rate Decay')
cmd:option('-weightDecay', 	0, 		'weight Decay')
cmd:option('-momentum', 	0, 		'momentum for sgd methods.')
cmd:text()

-- parse input params
params = cmd:parse(arg)
params.maindir = paths.cwd()

local scriptname = 'partCNN'
local rundir = cmd:string(scriptname, params, {lrate=true})
params.rundir = 'result/' .. rundir

if path_exists(params.rundir) then
   os.execute('rm -r ' .. params.rundir)
end
os.execute('mkdir -p ' .. params.rundir)
cmd:addTime(scriptname)
cmd:log(params.rundir .. '/log.txt', params)

params.trainLoss = torch.zeros(params.maxiter)
params.trainDw = torch.zeros(params.maxiter)
params.trainw = torch.zeros(params.maxiter)
params.cms = {}
starttime = sys.clock()
ProFi:start() 

logger = optim.Logger(params.rundir .. '/trainerrors')
logger:setNames{'training error'}

log_OA = optim.Logger(params.rundir .. '/OA')
log_OA:setNames{'Training OA', 'Validation OA'}

log_CA = optim.Logger(params.rundir .. '/CA')
log_CA:setNames{'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16'}

if params.cuda == 1 then
	cuda_enable = true
end
-- =============== END CMD LINE ============================================
if params.pixel == 0 then
	PatchSz = 1
	PatchPad = 0
else
	PatchSz = 9
	PatchPad = 1
end

if params.data == 1 then
	datastr = 'data/formatdata/Pines.mat';
	-- define sub spectral range
	-- ranges = {{1,40},{30,57},{50,78},{70,110},{100,150},{140,200}}
	-- ranges = {{1,40},{30,57},{50,78},{70,110},{100,150}}
	-- ranges = {{1,36},{36,80},{80,103},{103,200},{201,400}}
	ranges = {{1,36},{36,80},{80,103},{103,200},{1,200}}
	widthfull, exp_range = expandrange(ranges);
elseif params.data == 2 then
	datastr = 'data/formatdata/PaviaU.mat';
	-- ranges = {{1,40},{15,55},{20,65},{40,85},{60,95},{70,103}}
	-- ranges = {{1,40},{40,72},{72,103},{1,72},{104,206}}
	ranges = {{1,40},{40,72},{72,103},{1,72},{1,103}}
	widthfull, exp_range = expandrange(ranges);
elseif params.data == 3 then
	datastr = 'data/formatdata/Salinas.mat';
	ranges = {{1,40},{40,60},{60,80},{80,204},{1,204}}
	widthfull, exp_range = expandrange(ranges);
else
	assert(false, 'Data ID ranges from 1 to 3.');
end

train_data = mat.load(datastr,'data_train');
label_train = mat.load(datastr,'label_train');
train_label = label_train:reshape(label_train:size(1));
test_data = mat.load(datastr,'data_test');
label_test = mat.load(datastr,'label_test');
test_label = label_test:reshape(label_test:size(1)); 

params.numclasses = train_label:max()

params.subcnn = 5
model1 = loadmodelS1(PatchSz, ranges[1][1], ranges[1][2])
model2 = loadmodelS1(PatchSz, ranges[2][1], ranges[2][2])
model3 = loadmodelS1(PatchSz, ranges[3][1], ranges[3][2])
model4 = loadmodelS1(PatchSz, ranges[4][1], ranges[4][2])
model5 = loadmodelS1(PatchSz, ranges[5][1], ranges[5][2])
-- model6 = loadmodelS1(PatchSz, ranges[6][1], ranges[6][2], params.dp)

model7 = loadmodelS0(params.dp)
-- model7 = loadmodelS2(params.dp)

-- just for init weights, full model will be reloaded from file later. 
model = nn.Sequential()
model:add(model1)
model:add(model2)
model:add(model3)
model:add(model4)
model:add(model5)
-- model:add(model6)
model:add(model7)

-- sub models do not need so much epoches.
params.maxiter = params.maxiter/2

-- Init weights
if params.resetw == 1 then
	model = reset_weights(model, 'heuristic')
elseif params.resetw == 2 then
	model = reset_weights(model, 'xavier')
elseif params.resetw == 3 then
	model = reset_weights(model, 'xavier_caffe')
elseif params.resetw == 4 then 
	model = reset_weights(model, 'kaiming')
end

function getsubspectral(inputdata, ra, rb)
	local datacur = inputdata[{{},{},{},{ra, rb}}]:clone()
	return datacur
end

-- save result for stage 2
train_data_S = torch.zeros(train_data:size(1), params.numclasses*params.subcnn)
test_data_S = torch.zeros(test_data:size(1), params.numclasses*params.subcnn)
-- val_data_S = torch.zeros(val_data:size(1), params.numclasses*params.subcnn)

-- save normalized data for Full tune. 
-- One more padding dim is need for the split-join process.
train_data_Full = torch.zeros(train_data:size(1), 1, PatchSz, 1, widthfull)
test_data_Full = torch.zeros(test_data:size(1), 1, PatchSz, 1, widthfull)
-- val_data_Full = torch.zeros(val_data:size(1), 1, PatchSz, 1, widthfull)

-- Part CNN models
for i =  1, params.subcnn do
	modelcur = model.modules[i]
	local train_data_cur = getsubspectral(train_data, ranges[i][1], ranges[i][2])
	local test_data_cur = getsubspectral(test_data, ranges[i][1], ranges[i][2])
	-- local val_data_cur = getsubspectral(val_data, ranges[i][1], ranges[i][2])

	-- check if we want to normalize image
	if params.norm == 1 then -- whole dataset, DO not USE this version.
		train_data_cur, train_mean, train_std = normalizehyper(train_data_cur)
		test_data_cur = normalizehyper(test_data_cur, train_mean, train_std)
		-- val_data_cur = normalizehyper(val_data_cur, train_mean, train_std)

	elseif params.norm == 2 then  -- per instance
		train_data_cur = normalizehyperspec(train_data_cur)
		test_data_cur = normalizehyperspec(test_data_cur)
		-- val_data_cur = normalizehyperspec(val_data_cur)

	elseif params.norm == 3 then -- per spectral
		local sdim = train_data_cur:size(4)
		train_data_cur = normalpixel(train_data_cur:view(-1, sdim)):view(train_data_cur:size()):clone()
		test_data_cur = normalpixel(test_data_cur:view(-1, sdim)):view(test_data_cur:size()):clone()
		-- val_data_cur = normalpixel(val_data_cur:view(-1,sdim)):view(val_data_cur:size()):clone()
		
	elseif params.norm == 4 then -- ZCA whitening
		local tdr_vec = train_data_cur:view(-1,ranges[i][2]-ranges[i][1]+1):clone()
		local ter_vec = test_data_cur:view(-1,ranges[i][2]-ranges[i][1]+1):clone()
		-- local vdr_vec = val_data_cur:view(-1,ranges[i][2]-ranges[i][1]+1):clone()

		x,M,P = zca_whiten(tdr_vec);
		y,M,P = zca_whiten(ter_vec);
		-- z,M,P = zca_whiten(vdr_vec);

		train_data_cur = x:view(train_data_cur:size()):clone()
		test_data_cur = y:view(test_data_cur:size()):clone()
		-- val_data_cur = z:view(val_data_cur:size()):clone()
	end

	-- save normalized data for Full model tune.
	train_data_Full[{{},{},{},{},{exp_range[i][1], exp_range[i][2]}}]:copy(train_data_cur)
	test_data_Full[{{},{},{},{},{exp_range[i][1], exp_range[i][2]}}]:copy(test_data_cur)
	-- val_data_Full[{{},{},{},{},{exp_range[i][1], exp_range[i][2]}}]:copy(val_data_cur)

	-- train each sub model:
	local w,dl_dw = modelcur:getParameters()
	modelcur:training()
	print('Training sub model ' .. i .. ': ===================================================');
	_, opt_params = trainsub(modelcur, train_data_cur, train_label, train_data_cur, train_label)

	-- save training status parameters
	if params.optmsd == 1 or params.optmsd == 2 then
		if not OPT_params then 
			OPT_params = opt_params 
		else
			OPT_params.accDelta = torch.cat(OPT_params.accDelta, opt_params.accDelta)
			OPT_params.delta = torch.cat(OPT_params.delta, opt_params.delta)
			OPT_params.paramStd = torch.cat(OPT_params.paramStd, opt_params.paramStd)
			OPT_params.paramVariance = torch.cat(OPT_params.paramVariance, opt_params.paramVariance)
		end
	else
		print('opt-status are not going to be saved into file.')
	end

	-- save for stage 2
	modelcur:remove(#modelcur.modules) -- remove the softmax layer.

	modelcur:evaluate()
	local trdata = modelcur:forward(train_data_cur:cuda()):float()
	local tedata = modelcur:forward(test_data_cur:cuda()):float()
	-- local vldata = modelcur:forward(val_data_cur:cuda()):float()

	train_data_S[{{},{(i-1)*params.numclasses+1, i*params.numclasses}}]:copy(trdata)
	test_data_S[{{},{(i-1)*params.numclasses+1, i*params.numclasses}}]:copy(tedata)
	-- val_data_S[{{},{(i-1)*params.numclasses+1, i*params.numclasses}}]:copy(vldata)

	log_OA:add{0, 0}

end

-- Stage 2:: CNN model
-- train_data_S = train_data_S:view(train_data_S:size(1),1,1,train_data_S:size(2)):clone()
-- test_data_S = test_data_S:view(test_data_S:size(1),1,1,test_data_S:size(2)):clone()
-- val_data_S = val_data_S:view(val_data_S:size(1),1,1,val_data_S:size(2)):clone()

-- goto END
local w,dl_dw = model7:getParameters()
print('Training sub model 7: ====================================================')
model7:training()
_, opt_params = trainsub(model7, train_data_S, train_label, train_data_S, train_label)

model7:evaluate()
print('Testing sub model 7: ')
local confusionM = testcls(model7, test_data_S, test_label)

-- save training status paramets
if params.optmsd == 1 or params.optmsd == 2 then
	OPT_params.accDelta = torch.cat(OPT_params.accDelta, opt_params.accDelta)
	OPT_params.delta = torch.cat(OPT_params.delta, opt_params.delta)
	OPT_params.paramStd = torch.cat(OPT_params.paramStd, opt_params.paramStd)
	OPT_params.paramVariance = torch.cat(OPT_params.paramVariance, opt_params.paramVariance)
	-- OPT_params.learningRate = params.learningRate
end

-- check if we want to fine tune on the whole model.
if params.full == 1 then
	print('\n Fine-tuning the whole model: \n')
	params.maxiter = params.maxiter *2

	model = loadfull(exp_range)
	local w,dl_dw = model:getParameters()
	model:training()
	trainsub(model, train_data_Full, train_label, train_data_Full, train_label, nil, OPT_params)
	-- trainsub(model, train_data_Full, train_label, val_data_Full, val_label, nil, OPT_params)
else --model 7 is the best
	model = model7;
	train_data_Full = train_data_S;
	test_data_Full = test_data_S;
	-- val_data_Full = val_data_S;
end

print('Training time consuming ' .. (sys.clock() - starttime)/3600 .. ' h.')

model:evaluate()
print('Testing Whole model: ')
local confusionM = testcls(model, test_data_Full, test_label)

if cuda_enable then 
	model:cuda()
	test_data_Full = test_data_Full:cuda()
end
P = model:forward(test_data_Full)
P = P:float()
_, P = P:max(2)
mat.save(params.rundir .. '/predicts.mat', P)
mat.save('predicts.mat', P)

model:clearState()
torch.save(params.rundir .. '/model.net', model)
torch.save(params.rundir .. '/opt-status.t7', OPT_params)

-- save parameters
plot(params.trainLoss)
plot(params.trainDw, 'trainDw')
plot(params.trainw, 'trainw')

if #params.cms > 1 then
	local test_cm = torch.Tensor(params.cms)
	plot(test_cm, 'validation_accuracy')
end

logger:style{'-'}
logger:plot()    

log_OA:style{'-', '-'}
log_OA:plot()

log_CA:style{'-', '-','-', '-','-', '-','-', '-','-', '-','-', '-','-', '-','-', '-'}
log_CA:plot()

ProFi:stop()
ProFi:writeReport(params.rundir .. '/ProFireport.txt')

::END::