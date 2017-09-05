-- use BsNet as sub model. 
-- copy from model 6
-- add avg pooling
function loadmodel9(numclasses)
	-- N*8*1*2*1*Dim
	local model = load_BsNet()
	local p = nn.ParallelTable()
	p:add(model)
	p:add(model:clone('weight', 'bias', 'gradWeight','gradBias'))
	p:add(model:clone('weight', 'bias', 'gradWeight','gradBias'))
	p:add(model:clone('weight', 'bias', 'gradWeight','gradBias'))

	p:add(model:clone('weight', 'bias', 'gradWeight','gradBias'))
	p:add(model:clone('weight', 'bias', 'gradWeight','gradBias'))
	p:add(model:clone('weight', 'bias', 'gradWeight','gradBias'))
	p:add(model:clone('weight', 'bias', 'gradWeight','gradBias'))
	
	local modelx = nn.Sequential()
	modelx:add(nn.SplitTable(2))
	modelx:add(p)
	modelx:add(nn.JoinTable(2))
	modelx:add(nn.View(-1,8,1,numclasses))
	modelx:add(nn.VolumetricAveragePooling(8,1,1))  -- N*1*1*numclasses
	modelx:add(nn.View(-1,numclasses))
	modelx:add(nn.Linear(numclasses,numclasses))
	modelx:add(nn.Tanh())
	modelx:add(nn.Linear(numclasses,numclasses))
	modelx:add(nn.LogSoftMax())

	return modelx
end

function load_BsNet()
	-- data: N*1*2*1*dim

	-- define 5 submodels.
	params.subcnn = 5;

	ranges = {}
	if params.data == 1 then
		ranges = {{1,36},{37,80},{81,103},{104,200},{1,200}}
	elseif params.data == 2 then
		ranges = {{1,40},{40,72},{1,72},{72,103},{1,103}}
	else 
		ranges = {{1,40},{40,60},{60,80},{80,204},{1,204}}
	end

	local p = nn.ParallelTable()
	p:add(loadmodelS1(ranges[1][1], ranges[1][2]))
	p:add(loadmodelS1(ranges[2][1], ranges[2][2]))
	p:add(loadmodelS1(ranges[3][1], ranges[3][2]))
	p:add(loadmodelS1(ranges[4][1], ranges[4][2]))
	p:add(loadmodelS1(ranges[5][1], ranges[5][2]))

	local sj1 = nn.Sequential()
	sj1:add(nn.NarrowTable(ranges[1][1], ranges[1][2]-ranges[1][1]+1))
	sj1:add(nn.JoinTable(4))
	local sj2 = nn.Sequential()
	sj2:add(nn.NarrowTable(ranges[2][1], ranges[2][2]-ranges[2][1]+1))
	sj2:add(nn.JoinTable(4))
	local sj3 = nn.Sequential()
	sj3:add(nn.NarrowTable(ranges[3][1], ranges[3][2]-ranges[3][1]+1))
	sj3:add(nn.JoinTable(4))
	local sj4 = nn.Sequential()
	sj4:add(nn.NarrowTable(ranges[4][1], ranges[4][2]-ranges[4][1]+1))
	sj4:add(nn.JoinTable(4))
	local sj5 = nn.Sequential()
	sj5:add(nn.NarrowTable(ranges[5][1], ranges[5][2]-ranges[5][1]+1))
	sj5:add(nn.JoinTable(4))

	local c = nn.ConcatTable()
	c:add(sj1)
	c:add(sj2)
	c:add(sj3)
	c:add(sj4)
	c:add(sj5)

	modelS = loadmodelS0(params.dp)

	local model = nn.Sequential()
	model:add(nn.SplitTable(-1))
	model:add(c)
	model:add(p)
	model:add(nn.JoinTable(2))
	model:add(nn.PReLU())
	model:add(modelS)

	return model
end -- end of func.

function loadmodelS0(dropout)
	-- sub summation models
	local model = nn.Sequential()
	model:add(nn.Linear(params.numclasses*params.subcnn, 40))
	model:add(nn.PReLU())

	if dropout~=0 then
		model:add(nn.Dropout(dropout))
	end
	
	model:add(nn.Linear(40, params.numclasses))
	model:add(nn.PReLU())
	-- model:add(nn.LogSoftMax())

	return model
end

-- sub spectral model, input band ranges.
function loadmodelS1(ra, rb)
	-- N*1*2*Dim
	local kw = 7
	local kh1 = 2 -- pair dim == 2
	local kh2 = 1
	local numclasses = params.numclasses
	local dppcnt = params.dp

	local a = math.floor(((rb-ra+1) - (kw - 1))/2)
	local b = math.floor((a - (kw - 1)))
	local c = math.floor((b - (kw - 1)))
	local num_fmap = 20
	local fcinp = num_fmap*c

	local fcin = num_fmap*((rb-ra+1) - (kw - 1)* 3)

	local actFunc = nn.PReLU
	-- sub models
	local model = nn.Sequential()
	model:add(nn.SpatialConvolutionMM(1,num_fmap,kw,kh1))
	model:add(actFunc())
	model:add(nn.SpatialBatchNormalization(num_fmap, nil, nil, false))
	-- model:add(nn.SpatialMaxPooling(2,1))

	model:add(nn.SpatialConvolutionMM(num_fmap,num_fmap,kw,kh2))
	model:add(actFunc())
	model:add(nn.SpatialBatchNormalization(num_fmap, nil, nil, false))

	model:add(nn.SpatialConvolutionMM(num_fmap,num_fmap,kw,kh2))
	model:add(actFunc())
	model:add(nn.SpatialBatchNormalization(num_fmap, nil, nil, false))

	model:add(nn.Reshape(fcin))
	model:add(nn.Linear(fcin, 30))
	model:add(actFunc())

	if dppcnt ~= 0 then
		model:add(nn.Dropout(dppcnt))
	end

	model:add(nn.Linear(30, numclasses))
	model:add(actFunc())
	-- model:add(nn.LogSoftMax())

	return model
end	

function loadmodelS2() --bad
	-- sub models
	local kw = params.numclasses
	local kh1 = 1
	local kh2 = params.subcnn

	local fcin = params.subcnn

	local model = nn.Sequential()
	model:add(nn.View(-1,1,1,params.subcnn*params.numclasses))
	model:add(nn.SpatialConvolutionMM(1,1,kw,kh1,kw,kh1))
	model:add(nn.PReLU())

	-- model:add(nn.SpatialConvolutionMM(1,1,params.subcnn,kh1))
	-- model:add(nn.Linear(params.numclasses*params.subcnn, 50))
	
	model:add(nn.Reshape(fcin))
	model:add(nn.Linear(fcin, params.numclasses))
	model:add(nn.LogSoftMax())

	return model
end