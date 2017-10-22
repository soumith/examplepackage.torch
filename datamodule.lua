local datamodule = {}


------------------------------------------------------------------
-- Name: 	parallelize
-- Inputs: 	data array, targets array, ANN model, data array size
-- Outputs: new data array, new targest array, optimized batch size, new data size
-- Summary: This function receives elements from the user and calls the necessary
--			functions to prepare data parallelization across the MPI nodes.
--
------------------------------------------------------------------

function datamodule.parallelize( data, targets, model, size )

	local newdata, dataSize = datamodule.data_parallel(data, size)
	local newtargets = datamodule.data_parallel(targets, size)
	
	local speed = datamodule.comm_speed(data, targets, model)
	
	--print ("Communication time: " .. speed)
	
	
	-- determine optimal batch size
	local batchSize = datamodule.optimize_sync(speed, size)
	
	local momentumTensorReferences = {}
	--model.hooks.onBackward = function(state)
	--   print("hook caught")
	--end
	
	-- ensure all ranks are complete before returning
	mpi.barrier()
	
	return newdata, newtargets, batchSize, dataSize
end


------------------------------------------------------------------
-- Name: 	optimize_sync
-- Inputs: 	communication speed, data array size
-- Outputs: optimized batch size
-- Summary: This functions returns the optimized batch size based
--			on communication speed and data array size.
--
------------------------------------------------------------------

function datamodule.optimize_sync( speed, size )
	
	local batchSize = 100

	if (speed < 0.05 and size < 5000) then
		batchSize = 1
	elseif (speed < 0.1 and size < 10000) then
		batchSize = 10
	elseif (speed < 0.2 and size < 20000) then
		batchSize = 50
	end
	
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

function datamodule.data_parallel( data, size )

	--if (mpi.rank() == 0 ) then
		--print('Parallelizing Data')
	--end
	
	-- create local var for new dataset
	local newdata = {}
	
	-- determine which rank will get which data
	-- copy data from dataset to newdataset from start to end
	--local size      = #dataset.data
	local remainder = 0
	-- how many elements will be placed on each rank, remainder elements go to last rank
	local stripe = ( size - ( size % mpi.size() ) ) / mpi.size()
	if (mpi.rank() == mpi.size() - 1 ) then -- if I am the last rank
		remainder = size % mpi.size() 		-- only I get the remainder
	end
	size = size - remainder
	-- where will this rank's data start at
	local start  = ( mpi.rank() * stripe ) + 1
	-- where will this rank's data end at
	local finish = start + (stripe - 1) + remainder
	for t =  start,finish do 
		newdata[t - start + 1] = data[t]
	end
	
	--print ("Rank: " .. mpi.rank() .. " Start Point: " .. start .. " End Point: " .. finish .. " Stripe: " .. stripe .. " Remainder: " .. remainder)
	
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

function datamodule.comm_speed( data, targets, model )

	mpi.barrier() 
	
	--local start = os.date("%M%S") + os.clock()
	local timer = torch.Timer()
	timer:stop()
	timer:resume()
	
	function hook (why)
		if (debug.getinfo (2, "n").name == "backward") then
			print ("hook reached: ", why)
			print ("function =", debug.getinfo (2, "n").name)
		end
	end -- hook

	debug.sethook (hook, "r", 0)
	
	for i = 1,2 do 
		local output = model:forward(data[i])
		local df_do  = criterion:backward(output, targets[i])
		model:backward(data[i], df_do)
		mpinn.synchronizeGradients(model)
	end
	
	debug.sethook ()
	
	timer:stop()
	--local finish = os.date("%M%S") + os.clock()
	
	return (timer:time().real) / 10
end

return datamodule