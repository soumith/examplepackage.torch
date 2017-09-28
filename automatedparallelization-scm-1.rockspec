package = "automatedparallelization"
version = "scm-1"

source = {
   url = "git://github.com/ngrabaskas/automatedparallelization.torch",
   tag = "master"
}

description = {
   summary = "Automated Parallelization for Torch and TorchMPI",
   detailed = [[
   	    Automated Parallelization for Torch and TorchMPI
   ]],
   homepage = "https://github.com/ngrabaskas/automatedparallelization.torch"
}

dependencies = {
   "torch >= 7.0"
}

build = {
   type = "command",
   build_command = [[
cmake -E make_directory build;
cd build;
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)"; 
$(MAKE)
   ]],
   install_command = "cd build && $(MAKE) install"
}
