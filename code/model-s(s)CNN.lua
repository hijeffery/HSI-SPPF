-- Model0.lua
-- From : ACMMM2015: 
-- hyperspectral image classification with convolutional neural networks 

function loadMM15(patchsz, dim)
	-- patchsz expected value:  1 or 9
	assert(patchsz == 1 or patchsz == 9, 'Wrong Patch Size. Expected value: 1 or 9\n');
	assert(dim, 'Please give the length of hyperspectral bands.')
	local kw = 16
	local kh1 = patchsz or 9
	local kh2 = 1
	local numclasses = params.numclasses
	local fcin = 32*(dim - (kw - 1)* 3)

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

	model:add(nn.LogSoftMax())

	return model
end