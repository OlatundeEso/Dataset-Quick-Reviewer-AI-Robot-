If you are using other IDE's different from VS Code or Visual Studio, you may have issues with Wheel building while trying to install LangChain.
In that situation, what you can do is to
install Build Tools for Visual Studio 2017, select the workload “Visual C++ build tools” and check the options "C++/CLI support" and "VC++ 2015.3 v14.00 (v140) toolset for desktop" 

Another Issue Resolved:
If you are having challenges with the experimental codeline, simply install langchain-experimentalvis
pip install langchain-experimental