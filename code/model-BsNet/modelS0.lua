-- modelS0.lua

function loadmodelS0(dropout)
	-- sub models
	local model = nn.Sequential()
	model:add(nn.Linear(params.numclasses*params.subcnn, 50))
	model:add(nn.Tanh())

	if dropout~=0 then
		model:add(nn.Dropout(dropout))
	end
	
	model:add(nn.Linear(50, params.numclasses))

	model:add(nn.LogSoftMax())

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
	model:add(nn.Tanh())

	-- model:add(nn.SpatialConvolutionMM(1,1,params.subcnn,kh1))
	-- model:add(nn.Linear(params.numclasses*params.subcnn, 50))
	
	model:add(nn.Reshape(fcin))
	model:add(nn.Linear(fcin, params.numclasses))
	model:add(nn.LogSoftMax())

	return model
end