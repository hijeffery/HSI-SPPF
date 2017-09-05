-- use model from MM15
-- modify model 3
-- use volumetricavgpool 
function loadmodel7(numclasses, bandnum)
	-- data: N*8*1*2*dim
	local kw = 16
	local kh1 = params.groupsz or 2
	local kh2 = 1
	local fcin = 32*(bandnum - (kw - 1)* 3)

	local actFunc = nn.Tanh

	model = nn.Sequential()
	model:add(nn.SpatialConvolutionMM(1,32,kw,kh1))
	model:add(actFunc())

	model:add(nn.SpatialConvolutionMM(32,32,kw,kh2))
	model:add(actFunc())

	model:add(nn.SpatialConvolutionMM(32,32,kw,kh2))
	model:add(actFunc())

	model:add(nn.Reshape(fcin))
	model:add(nn.Linear(fcin, 400))
	model:add(actFunc())

	model:add(nn.Linear(400, 200))
	model:add(actFunc())

	model:add(nn.Linear(200, numclasses))
	model:add(actFunc())
	model:add(nn.View(-1,1,1,numclasses))

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
	modelx:add(nn.JoinTable(2)) -- N*8*1*dim
	modelx:add(nn.VolumetricAveragePooling(8,1,1))  -- N*1*1*numclasses
	modelx:add(nn.View(-1,numclasses))
	modelx:add(nn.Tanh())
	modelx:add(nn.Linear(numclasses,numclasses))
	modelx:add(nn.Tanh())
	modelx:add(nn.Linear(numclasses,numclasses))
	modelx:add(nn.LogSoftMax())

	return modelx
end -- end of func.