namespace kernel
{
	class model {
	private:
		double	defval,
				cover,
				impulse,
				neupeak; 

		std::string name;
		std::vector<std::vector<double>> layers;
		std::vector<std::vector<unsigned int>> linking;
		std::vector<std::vector<std::vector<std::vector<double>>>> weights;
		std::vector<std::vector<std::vector<std::vector<double>>>> conducts;
	public:
		model ( std::vector<std::pair<unsigned int, std::vector<unsigned int>>> = {}, 
				double = 1, double = 1, double = 1, double = 1, std::string = "unit" );

		void setIn (std::vector<double>, int, int);
		std::vector<double> getOut (void);

		void dropBuffers (void);
		void dropOut (void);

		void dropPart (double = 1);
		void exec (bool = false, double = 0);
	};
}
