// execution iteration
void kernel::model::exec (bool is_training, double motivator)
{
	double sumx;
	arma::Mat<double> trans;
	std::vector<double> sumtrans;
	double target;

	for (uint32_t x_lay = 0; x_lay < layers.size(); ++x_lay) { 
		for (uint32_t y_lay = 0; y_lay < linking[x_lay].size(); ++y_lay) {  

		// precalcutations
		sumx = 0; // sum of an x's vals (in x layer) as EX
		for (uint32_t x_in = 0; x_in < layers[x_lay].size(); ++x_in) {
			sumx += layers[x_lay][x_in];
		}

		trans = arma::Mat<double>(layers[x_lay].size(), layers[linking[x_lay][y_lay]].size());
		sumtrans = std::vector<double>(layers[linking[x_lay][y_lay]].size(), 0);

		// collecting transmitted signal ------------------------------------->
		for (uint32_t x_in = 0; x_in < layers[x_lay].size(); ++x_in) {
			for (uint32_t y_in = 0; y_in < layers[linking[x_lay][y_lay]].size(); ++y_in) {
				trans.at(x_in, y_in) =
				/*   x   */layers[x_lay][x_in] * 
				/*   y   */layers[linking[x_lay][y_lay]][y_in] * 
				/*   S   */weights[x_lay][y_lay].at(x_in, y_in) *
				/*   D   */conducts[x_lay][y_lay].at(x_in, y_in) *
				/*   d   */((1 - layers[linking[x_lay][y_lay]][y_in] / neupeak) / (sumx / layers[x_lay][x_in])) *
				/*   e   */impulse;

				sumtrans[y_in] += trans.at(x_in, y_in);
			}
		}

		// changing weights  ------------------------------------------------->
		if (is_training) {
			for (int y_in = 0; y_in < layers[linking[x_lay][y_lay]].size(); ++y_in) {
				for (int x_in = 0; x_in < layers[x_lay].size(); ++x_in) {
					// ----------------------------------------------
					target = trans.at(x_in, y_in) / sumtrans[y_in];

					if (target > weights[x_lay][y_lay].at(x_in, y_in))
						weights[x_lay][y_lay].at(x_in, y_in) += (target - weights[x_lay][y_lay].at(x_in, y_in)) * motivator;
					else 
						weights[x_lay][y_lay].at(x_in, y_in) -= (weights[x_lay][y_lay].at(x_in, y_in) - target) * motivator;
					// ----------------------------------------------

					// if weight will become in small value
					if (weights[x_lay][y_lay].at(x_in, y_in) < 0 || std::isnan(weights[x_lay][y_lay].at(x_in, y_in)))
						weights[x_lay][y_lay].at(x_in, y_in) = 0;

					printf("%lf ", weights[x_lay][y_lay].at(x_in, y_in));
				}
				puts("");
			}
		}

		// transform values of x layers and y layers -------------------------->
		for (uint32_t x_in = 0; x_in < layers[x_lay].size(); ++x_in) 
			for (uint32_t y_in = 0; y_in < layers[linking[x_lay][y_lay]].size(); ++y_in) {
				layers[x_lay][x_in] -= trans.at(x_in, y_in);
				layers[linking[x_lay][y_lay]][y_in] += trans.at(x_in, y_in);
			}

		// activation function ---------------------------------------------->
		// ------------------------------------------------------------------>
		buf_d[0] = 0;
		for (uint32_t y_in = 0; y_in < layers[linking[x_lay][y_lay]].size(); ++y_in)
			if (layers[linking[x_lay][y_lay]][y_in] < buf_d[0]) buf_d[0] = layers[linking[x_lay][y_lay]][y_in];

		buf_d[1] = 0;
		for (uint32_t y_in = 0; y_in < layers[linking[x_lay][y_lay]].size(); ++y_in)
			if (layers[linking[x_lay][y_lay]][y_in] > buf_d[1]) buf_d[1] = layers[linking[x_lay][y_lay]][y_in];

		for (uint32_t y_in = 0; y_in < layers[linking[x_lay][y_lay]].size(); ++y_in) {
			layers[linking[x_lay][y_lay]][y_in] = (layers[linking[x_lay][y_lay]][y_in] - buf_d[0]) / (buf_d[1] - buf_d[0]);

			if (layers[linking[x_lay][y_lay]][y_in] > 0.5) 
				layers[linking[x_lay][y_lay]][y_in] -= pow(double(1) - layers[linking[x_lay][y_lay]][y_in], 2); 
			else 
				layers[linking[x_lay][y_lay]][y_in] += pow(layers[linking[x_lay][y_lay]][y_in], 2); 

			if (layers[linking[x_lay][y_lay]][y_in] < default_v || std::isnan(layers[linking[x_lay][y_lay]][y_in])) 
				layers[linking[x_lay][y_lay]][y_in] = default_v;
		}

		// ------------------------------------------------------------------->
		buf_d[0] = 0;
		for (uint32_t x_in = 0; x_in < layers[x_lay].size(); ++x_in)
			if (layers[x_lay][x_in] < buf_d[0]) buf_d[0] = layers[x_lay][x_in];

		buf_d[1] = 0;
		for (uint32_t x_in = 0; x_in < layers[x_lay].size(); ++x_in)
			if (layers[x_lay][x_in] > buf_d[1]) buf_d[1] = layers[x_lay][x_in];

		for (uint32_t x_in = 0; x_in < layers[x_lay].size(); ++x_in) {
			layers[x_lay][x_in] = (layers[x_lay][x_in] - buf_d[0]) / (buf_d[1] - buf_d[0]);

			if (layers[x_lay][x_in] > 0.5) 
				layers[x_lay][x_in] -= pow(double(1) - layers[x_lay][x_in], 2); 
			else 
				layers[x_lay][x_in] += pow(layers[x_lay][x_in], 2); 

			if (layers[x_lay][x_in] < default_v || std::isnan(layers[x_lay][x_in])) 
				layers[x_lay][x_in] = default_v;
			}
		}
	}
}
