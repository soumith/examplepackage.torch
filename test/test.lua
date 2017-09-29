require 'nn'

example = require 'automatedparallelization'

model = nn.Sequential()
-- stage 1 : mean suppresion -> filter bank -> squashing -> max pooling
model:add(nn.SpatialConvolutionMM(3, 32, 5, 5))
model:add(nn.Tanh())
model:add(nn.SpatialMaxPooling(3, 3, 3, 3, 1, 1))
-- stage 2 : mean suppresion -> filter bank -> squashing -> max pooling
model:add(nn.SpatialConvolutionMM(32, 64, 5, 5))
model:add(nn.Tanh())
model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
-- stage 3 : standard 2-layer MLP:
model:add(nn.Reshape(64*3*3))
model:add(nn.Linear(64*3*3, 200))
model:add(nn.Tanh())
model:add(nn.Linear(200, 10))

trainSize = 10

dataset = {
	data = {}
	}

for i = 0,trainSize do
	dataset.data[i] = i + 1
end

print ('--------Testing Data Module---------------')
arg1, arg2 = example.datamodule.parallelize( dataset ) 
print ('returned values:')
print ('batchSize = ', arg1)
print ('dataset size = ', arg2)
print ('\n')

print ('--------Testing Node Module---------------')
example.nodemodule.parallelize( model )
