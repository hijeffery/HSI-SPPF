-- Model0.lua
-- From : TGRS16: 
-- HSI using deep pixel-pair features

function loadmodel0(final_conv_k, num_classes)
	if not final_conv_k then
		final_conv_k = 13;
	end

	if not num_classes then
		num_classes = params.numclasses;
	end

	local model = nn.Sequential()
	model:add(nn.SpatialConvolutionMM(1,10,9,1))
	model:add(nn.ReLU())

	model:add(nn.SpatialConvolutionMM(10,10,1,2))
	model:add(nn.ReLU())

	model:add(nn.SpatialConvolutionMM(10,10,3,1,1,1,1,0))
	model:add(nn.ReLU())

	model:add(nn.SpatialMaxPooling(3,1))

	model:add(nn.SpatialConvolutionMM(10,20,3,1))
	model:add(nn.ReLU())

	model:add(nn.SpatialConvolutionMM(20,20,3,1))
	model:add(nn.ReLU())

	model:add(nn.SpatialMaxPooling(2,1))

	model:add(nn.SpatialConvolutionMM(20,40,3,1))
	model:add(nn.ReLU())

	model:add(nn.SpatialConvolutionMM(40,40,3,1))
	model:add(nn.ReLU())

	model:add(nn.SpatialMaxPooling(2,1))

	model:add(nn.SpatialConvolutionMM(40,80,final_conv_k,1))
	model:add(nn.ReLU())

	model:add(nn.Reshape(80))
	model:add(nn.Linear(80, 80))
	model:add(nn.ReLU())

	model:add(nn.Linear(80, num_classes))

	model:add(nn.LogSoftMax())

	return model
end