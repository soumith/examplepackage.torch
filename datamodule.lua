local datamodule = {}

function datamodule.parallelize( dataset )
	print('Data Parallelized!')
	print (dataset)
	
	local batchSize = 100
	local dataSize = #dataset.data
	
	return batchSize, dataSize
end

return datamodule