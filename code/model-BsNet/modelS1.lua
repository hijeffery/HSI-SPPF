-- model10.lua
-- sub spectral model

function loadmodelS1(patchsz, ra, rb)
	-- patchsz expected value:  1 or 9
	assert(patchsz == 1 or patchsz == 9, "\n patch size expects to be 1 or 9. \n")

	local kw = 7
	local kh1 = patchsz
	local kh2 = 1
	local numclasses = params.numclasses
	local dppcnt = params.dp

	local a = math.floor(((rb-ra+1) - (kw - 1))/2)
	local b = math.floor((a - (kw - 1)))
	local c = math.floor((b - (kw - 1)))
	local num_fmap = 20
	local fcinp = num_fmap*c

	local fcin = num_fmap*((rb-ra+1) - (kw - 1)* 3)

	-- sub models
	local model = nn.Sequential()
	model:add(nn.SpatialConvolutionMM(1,num_fmap,kw,kh1))
	model:add(nn.Tanh())
	-- model:add(nn.SpatialBatchNormalization(num_fmap, nil, nil, false))
	-- model:add(nn.SpatialMaxPooling(2,1))

	model:add(nn.SpatialConvolutionMM(num_fmap,num_fmap,kw,kh2))
	model:add(nn.Tanh())
	-- model:add(nn.SpatialBatchNormalization(num_fmap, nil, nil, false))

	model:add(nn.SpatialConvolutionMM(num_fmap,num_fmap,kw,kh2))
	model:add(nn.Tanh())
	-- model:add(nn.SpatialBatchNormalization(num_fmap, nil, nil, false))

	model:add(nn.Reshape(fcin))
	model:add(nn.Linear(fcin, 50))
	model:add(nn.Tanh())

	if dppcnt ~= 0 then
		model:add(nn.Dropout(dppcnt))
	end

	model:add(nn.Linear(50, numclasses))
	model:add(nn.LogSoftMax())

	return model
end	