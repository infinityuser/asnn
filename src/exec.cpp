using namespace std;

// execution iteration
void kernel::model::exec (bool is_training, double motivator)
{
	vector<vector<double>> trans;
	vector<double> sumtransy, sumtransx;
	
	double target;
	double buf_d[2];
	double sumx;

	for (int x_lay = 0; x_lay < layers.size(); ++x_lay) { 
		for (int y_lay = 0; y_lay < linking[x_lay].size(); ++y_lay) {  

		// precalcutations
		sumx = 0; // sum of an x's vals (in x layer) as EX
		for (int x_in = 0; x_in < layers[x_lay].size(); ++x_in) {
			sumx += layers[x_lay][x_in];
		}

		trans = vector<vector<double>>(layers[x_lay].size(), (vector<double>(layers[linking[x_lay][y_lay]].size(), 0)));
		sumtransy = vector<double>(layers[linking[x_lay][y_lay]].size(), 0);
		sumtransx = vector<double>(layers[x_lay].size(), 0);

		// collecting transmitted signal ------------------------------------->
		for (int x_in = 0; x_in < layers[x_lay].size(); ++x_in) {
			for (int y_in = 0; y_in < layers[linking[x_lay][y_lay]].size(); ++y_in) {
				trans[x_in][y_in] =
				/*   x   */layers[x_lay][x_in] * 
				/*   y   */layers[linking[x_lay][y_lay]][y_in] * 
				/*   S   */(weights[x_lay][y_lay][x_in][y_in] ? weights[x_lay][y_lay][x_in][y_in] : defval) *
				/*   D   */conducts[x_lay][y_lay][x_in][y_in] *
				/*   d   */((1 - layers[linking[x_lay][y_lay]][y_in] / neupeak) / (sumx / layers[x_lay][x_in])) *
				/*   n   */impulse;

				sumtransy[y_in] += trans[x_in][y_in];
				sumtransx[x_in] += trans[x_in][y_in];
			}
		}

		// changing weights  ------------------------------------------------->
		if (is_training) {
			for (int y_in = 0; y_in < layers[linking[x_lay][y_lay]].size(); ++y_in) {
				for (int x_in = 0; x_in < layers[x_lay].size(); ++x_in) {
					// ----------------------------------------------
					// S = S + (t - S) * m
					target = (trans[x_in][y_in] / sumtransy[y_in] + trans[x_in][y_in] / sumtransx[x_in]) / 2;
					weights[x_lay][y_lay][x_in][y_in] += (target - weights[x_lay][y_lay][x_in][y_in]) * motivator;
					// ----------------------------------------------

					// if there will be some troubles with in double operations
					if (weights[x_lay][y_lay][x_in][y_in] < 0 || isnan(weights[x_lay][y_lay][x_in][y_in]))
						weights[x_lay][y_lay][x_in][y_in] = 0;
				}
			}
		}

		// transform values of x layers and y layers -------------------------->
		for (int x_in = 0; x_in < layers[x_lay].size(); ++x_in) 
			for (int y_in = 0; y_in < layers[linking[x_lay][y_lay]].size(); ++y_in) {
				layers[x_lay][x_in] -= trans[x_in][y_in];
				layers[linking[x_lay][y_lay]][y_in] += trans[x_in][y_in];
			}

		// activation function ---------------------------------------------->
		// ------------------------------------------------------------------>
		buf_d[0] = 0;
		for (int y_in = 0; y_in < layers[linking[x_lay][y_lay]].size(); ++y_in)
			if (layers[linking[x_lay][y_lay]][y_in] < buf_d[0]) buf_d[0] = layers[linking[x_lay][y_lay]][y_in];

		buf_d[1] = 0;
		for (int y_in = 0; y_in < layers[linking[x_lay][y_lay]].size(); ++y_in)
			if (layers[linking[x_lay][y_lay]][y_in] > buf_d[1]) buf_d[1] = layers[linking[x_lay][y_lay]][y_in];

		for (int y_in = 0; y_in < layers[linking[x_lay][y_lay]].size(); ++y_in) {
			layers[linking[x_lay][y_lay]][y_in] = (layers[linking[x_lay][y_lay]][y_in] - buf_d[0]) / (buf_d[1] - buf_d[0]) * neupeak;

			if (layers[linking[x_lay][y_lay]][y_in] > neupeak / 2) 
				layers[linking[x_lay][y_lay]][y_in] -= pow(double(1) - layers[linking[x_lay][y_lay]][y_in], 2); 
			else 
				layers[linking[x_lay][y_lay]][y_in] += pow(layers[linking[x_lay][y_lay]][y_in], 2); 

			if (layers[linking[x_lay][y_lay]][y_in] < defval || isnan(layers[linking[x_lay][y_lay]][y_in])) 
				layers[linking[x_lay][y_lay]][y_in] = defval;
		}

		// ------------------------------------------------------------------->
		buf_d[0] = 0;
		for (int x_in = 0; x_in < layers[x_lay].size(); ++x_in)
			if (layers[x_lay][x_in] < buf_d[0]) buf_d[0] = layers[x_lay][x_in];

		buf_d[1] = 0;
		for (int x_in = 0; x_in < layers[x_lay].size(); ++x_in)
			if (layers[x_lay][x_in] > buf_d[1]) buf_d[1] = layers[x_lay][x_in];

		for (int x_in = 0; x_in < layers[x_lay].size(); ++x_in) {
			layers[x_lay][x_in] = (layers[x_lay][x_in] - buf_d[0]) / (buf_d[1] - buf_d[0]) * neupeak;

			if (layers[x_lay][x_in] > neupeak / 2) 
				layers[x_lay][x_in] -= pow(double(1) - layers[x_lay][x_in], 2); 
			else 
				layers[x_lay][x_in] += pow(layers[x_lay][x_in], 2); 

			if (layers[x_lay][x_in] < defval || isnan(layers[x_lay][x_in])) 
				layers[x_lay][x_in] = defval;
			}
		}
	}
}
