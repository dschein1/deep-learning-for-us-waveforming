pe = pyenv('Version','C:\Users\DrorSchein\miniconda3\envs\deep_env\python.exe')
if count(py.sys.path,'') == 0
    insert(py.sys.path,int32(0),'');
end
pyExec = 'C:\Users\DrorSchein\miniconda3\envs\deep_env\python.exe';
pyRoot = fileparts(pyExec);
p = getenv('PATH');
p = strsplit(p, ';');
addToPath = {
    pyRoot
    fullfile(pyRoot, 'Library', 'mingw-w64', 'bin')
    fullfile(pyRoot, 'Library', 'usr', 'bin')
    fullfile(pyRoot, 'Library', 'bin')
    fullfile(pyRoot, 'Scripts')
    fullfile(pyRoot, 'bin')
    };
p = [addToPath(:); p(:)];
p = unique(p, 'stable');
p = strjoin(p, ';');
setenv('PATH', p);
%%
lib = py.importlib.import_module('matlabInterface')
interface = lib
