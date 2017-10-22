require 'nn'

-- Automated Parallelization
example = require 'automatedparallelization'


-- TorchMPI
require 'torchmpi'
mpi = require('torchmpi')
mpi.start(true)  --true equals use GPU


-- Set CNN 
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

criterion = nn.ClassNLLCriterion()

-- TorchMPI
mpinn = require('torchmpi.nn')
mpinn.synchronizeParameters(model)

-- Set Sample Dataset
----------------------------------------------------------------------
-- get/create dataset
--
trsize = 100
tesize = 25

-- load dataset
trainData = {
   data = torch.Tensor(50000, 3072),
   labels = torch.Tensor(50000),
   size = function() return trsize end
}
for i = 0,4 do
   subset = torch.load('/home/ubuntu/demos-master/train-on-cifar/cifar-10-batches-t7/data_batch_' .. (i+1) .. '.t7', 'ascii')
   trainData.data[{ {i*10000+1, (i+1)*10000} }] = subset.data:t()
   trainData.labels[{ {i*10000+1, (i+1)*10000} }] = subset.labels
end
trainData.labels = trainData.labels + 1

subset = torch.load('/home/ubuntu/demos-master/train-on-cifar/cifar-10-batches-t7/test_batch.t7', 'ascii')
testData = {
   data = subset.data:t():double(),
   labels = subset.labels[1]:double(),
   size = function() return tesize end
}
testData.labels = testData.labels + 1

-- resize dataset (if using small version)
trainData.data = trainData.data[{ {1,trsize} }]
trainData.labels = trainData.labels[{ {1,trsize} }]

testData.data = testData.data[{ {1,tesize} }]
testData.labels = testData.labels[{ {1,tesize} }]

-- reshape data
trainData.data = trainData.data:reshape(trsize,3,32,32)
testData.data = testData.data:reshape(tesize,3,32,32)

-- Test Data Module
if (mpi.rank() == 0) then
	print ('--------Testing Data Module---------------')
end

mpi.barrier()

trainData.data, trainData.labels, batchSize, newSize = example.datamodule.parallelize( trainData.data, trainData.labels, model, trsize ) 
if (mpi.rank() == 0) then
	print ('returned values:')
	print ('batchSize = ', batchSize)
	print ('dataset size = ', newSize)
end

-- print(dataset.data[1])

-- Test Node Model
--print ('--------Testing Node Module---------------')
--example.nodemodule.parallelize( model )
