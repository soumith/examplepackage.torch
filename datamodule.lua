local datamodule = {}


------------------------------------------------------------------
-- Name: 	parallelize
-- Inputs: 	data array, targets array, ANN model, data array size, mpi, mpinn, batchSize
-- Outputs: new data array, new targest array, optimized batch size, new data size
-- Summary: This function receives elements from the user and calls the necessary
--			functions to prepare data parallelization across the MPI nodes.
--
------------------------------------------------------------------

function datamodule.parallelize( data, targets, model, size, mpi_obj, mpinn_obj, batchSize )
	
	-- check required parameters were passed
	if (data == nil or targets == nil or model == nil or size == nil) then
		print ("-usage for parallelize(data, targets, model, size)")
		return -1
	end
	
	-- if no MPI object is passed create one
	if (mpi_obj == nil) then
		require 'torchmpi'
		mpi_obj = require('torchmpi')
		mpi_obj.start(true)  --true equals use GPU
	end
	
	-- if no MPI NN object is passed create one
	if (mpinn_obj == nil) then
		mpinn_obj = require('torchmpi.nn')
		mpinn_obj.synchronizeParameters(model)
	end
	
	--mpi_obj.barrier()
	
	-- split data and targets across all nodes
	local newdata, dataSize = datamodule.data_parallel(data, size, mpi_obj)
	local newtargets = datamodule.data_parallel(targets, size, mpi_obj)
	
	-- determine speed
	local speed = datamodule.comm_speed(data, targets, model, mpi_obj, mpinn_obj)
	
	-- determine optimal batch size
	datamodule.optimize_sync(speed, dataSize, model, mpinn_obj, mpi_obj, batchSize)
	
	-- ensure all ranks are complete before returning
	mpi_obj.barrier()
	
	return newdata, newtargets, dataSize
end


------------------------------------------------------------------
-- Name: 	optimize_sync
-- Inputs: 	communication speed, data array size
-- Outputs: optimized batch size
-- Summary: This functions returns the optimized batch size based
--			on communication speed and data array size.
--
------------------------------------------------------------------

function datamodule.optimize_sync( speed, size, model, mpinn, mpi, batchSize )
	
	-- if batchSize is not given it will be optimized
	if (batchSize == nil) then
		batchSize = 100

		if (speed < 0.05 and size < 1000) then
			batchSize = 1
		elseif (speed < 0.1 and size < 2500) then
			batchSize = 10
		elseif (speed < 0.2 and size < 5000) then
			batchSize = 50
		end
	end
	
	--
	-- After comm test and batchSize determined override backward propogation function to syncGradients
	--
	function nn.Sequential:backward(input, gradOutput, scale)
		scale = scale or 1
		local currentGradOutput = gradOutput
		local currentModule = self.modules[#self.modules]
		for i=#self.modules-1,1,-1 do
			local previousModule = self.modules[i]
			currentGradOutput = self:rethrowErrors(currentModule, i+1, 'backward', previousModule.output, currentGradOutput, scale)
			currentModule.gradInput = currentGradOutput
			currentModule = previousModule
		end
		currentGradOutput = self:rethrowErrors(currentModule, 1, 'backward', input, currentGradOutput, scale)
		self.gradInput = currentGradOutput
		
		-- additional sync functionality added
		-- * 
		if (self.sync_counter == nil) then
			self.sync_counter = 1;
		end
		
		-- sync and shutdown when dataset is complete
		if (self.sync_counter == size) then
			print("Rank: " .. mpi.rank() .. " training complete");
			mpinn.synchronizeGradients(model)
			mpi.stop()
		elseif (self.sync_counter % batchSize == 0) then -- sync when batch is complete 
			print("Rank: " .. mpi.rank() .. " backward batch complete, count: " .. self.sync_counter);
			mpinn.synchronizeGradients(model)
		end
		self.sync_counter = self.sync_counter + 1
		-- *
		-- end of additional functionality
		
		return currentGradOutput
	end
	
	--[[ This function needs to be overriden to allow for stochastic gradient training
	function nn.Module:accUpdateGradParameters(input, gradOutput, lr)
	   if self.shared then
		  self:sharedAccUpdateGradParameters(input, gradOutput, lr)
	   else
		  self:defaultAccUpdateGradParameters(input, gradOutput, lr)
	   end
	   print("override")
	end
	]]--
	
	return batchSize
end


------------------------------------------------------------------
-- Name: 	data_parallel
-- Inputs: 	array, array size
-- Outputs: new array, new size
-- Summary: This function splits the data array evenly across the MPI nodes.
--			Any remainder is given to the last node.
--
------------------------------------------------------------------

function datamodule.data_parallel( data, size, mpi )
	
	-- create local var for new dataset
	local newdata = {}
	
	-- determine which rank will get which data
	-- copy data from dataset to newdataset from start to end
	local remainder = 0
	-- how many elements will be placed on each rank, remainder elements go to last rank
	local stripe = ( size - ( size % mpi.size() ) ) / mpi.size()
	--if (mpi.rank() == mpi.size() - 1 ) then -- if I am the last rank
	--	remainder = size % mpi.size() 		-- only I get the remainder
	--end
	size = size - remainder
	-- where will this rank's data start at
	local start  = ( mpi.rank() * stripe ) + 1
	-- where will this rank's data end at
	local finish = start + (stripe - 1) + remainder
	for t =  start,finish do 
		newdata[t - start + 1] = data[t]
	end
	
	print ("Rank: " .. mpi.rank() .. " Start Point: " .. start .. " End Point: " .. finish .. " Stripe: " .. stripe .. " Remainder: " .. remainder)
	
	-- return new dataset size
	local dataSize = stripe + remainder
	
	return newdata, dataSize
end
	

------------------------------------------------------------------
-- Name: 	comm_speed
-- Inputs: 	data array, target array, ANN model
-- Outputs: communication speed
-- Summary:	This function runs ten test of the network forward and backward
--			propogation with a sync of gradients across nodes. This is done
--			10 times and the average time returned.
--
------------------------------------------------------------------

function datamodule.comm_speed( data, targets, model, mpi, mpinn )

	-- ensure all ranks are ready before beginning comm check
	mpi.barrier() 
	
	--local start = os.date("%M%S") + os.clock()
	local timer = torch.Timer()
	timer:stop()
	timer:resume()
	
	for i = 1,10 do 
		local output = model:forward(data[i])
		local df_do  = criterion:backward(output, targets[i])
		model:backward(data[i], df_do)
		mpinn.synchronizeGradients(model)
	end
	
	timer:stop()
	
	local speed = (timer:time().real) / 10
	
	print("Rank: " .. mpi.rank() .. " comm speed: " .. speed)
	
	return speed
end

return datamodule