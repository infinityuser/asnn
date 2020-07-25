namespace kernel
{
    class model {
		private:
            std::vector<double> buffer_vec;
            double buffer_T;
            arma::Mat<double> buf_mat;
            int32_t buf_ui[8];
            double buf_d[4];

            double default_v = 1; 
            double modify = 1;
			double impulse = 1;
			double neupeak = 1; 

            std::string name = "";
            std::vector<std::vector<double>> layers;
            std::vector<std::vector<uint32_t>> linking;
            std::vector<std::vector<arma::Mat<double>>> weights;
            std::vector<std::vector<arma::Mat<double>>> conducts;
		public:
            model (std::vector<std::pair<unsigned int, std::vector<unsigned int>>>, double, double, double, double, std::string);
            model (std::string = "");
            
			void backup (std::string);
            void open (std::string);
            void getConfFile (std::string);
            
			void setIn (std::vector<double>, int, int);
            std::vector<double> getOut (void);
			
			void dropBuffers (void);
			void dropOut (void);
			void dropPart (double);
            void exec (bool, double);
    };
};
