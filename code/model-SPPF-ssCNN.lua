-- use model from MM15
-- modify model 0, add avgpool layer
function loadmodel8(numclasses, bandnum)
	-- data: N*8*2*dim
	local kw = 16
	local kh1 = 2
	local kh2 = 1
	local fcin = 32*(bandnum - (kw - 1)* 3)

	model = nn.Sequential()
	model:add(nn.SpatialConvolutionMM(1,32,kw,kh1))
	model:add(nn.Tanh())

	model:add(nn.SpatialConvolutionMM(32,32,kw,kh2))
	model:add(nn.Tanh())

	model:add(nn.SpatialConvolutionMM(32,32,kw,kh2))
	model:add(nn.Tanh())

	model:add(nn.Reshape(fcin))
	model:add(nn.Linear(fcin, 800))
	model:add(nn.Tanh())

	model:add(nn.Linear(800, 800))
	model:add(nn.Tanh())

	model:add(nn.Linear(800, numclasses))
	model:add(nn.Tanh())
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
	modelx:add(nn.JoinTable(2))
	modelx:add(nn.VolumetricAveragePooling(8,1,1))  -- N*1*1*numclasses
	modelx:add(nn.View(-1,numclasses))
	modelx:add(nn.Tanh())
	modelx:add(nn.Linear(numclasses,numclasses))
	modelx:add(nn.Tanh())
	modelx:add(nn.Linear(numclasses,numclasses))
	modelx:add(nn.LogSoftMax())

	return modelx
end -- end of func.