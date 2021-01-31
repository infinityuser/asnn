using namespace std;

// execution iteration
void kernel::model::exec (bool is_training, double motivator)
{
	vector<vector<double>> trans;
	vector<double> maxtransy, maxtransx, avetransx, avetransy, mintransy, mintransx;
	
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
		maxtransy = vector<double>(layers[linking[x_lay][y_lay]].size(), -neupeak);
		maxtransx = vector<double>(layers[x_lay].size(), -neupeak);
		mintransy = vector<double>(layers[linking[x_lay][y_lay]].size(), neupeak);
		mintransx = vector<double>(layers[x_lay].size(), neupeak);
		avetransy = vector<double>(layers[linking[x_lay][y_lay]].size(), 0);
		avetransx = vector<double>(layers[x_lay].size(), 0);

		// collecting transmitted signal ------------------------------------->
		for (int x_in = 0; x_in < layers[x_lay].size(); ++x_in) {
			for (int y_in = 0; y_in < layers[linking[x_lay][y_lay]].size(); ++y_in) {
				trans[x_in][y_in] =
				/*   x   */layers[x_lay][x_in] * 
				/*   y   */layers[linking[x_lay][y_lay]][y_in] * //just to check hypothesis
				/*   w   */(weights[x_lay][y_lay][x_in][y_in] ? weights[x_lay][y_lay][x_in][y_in] : defval) *
				/*   d   */conducts[x_lay][y_lay][x_in][y_in] *
				/* delta *///((1 - layers[linking[x_lay][y_lay]][y_in] / neupeak) / (sumx / layers[x_lay][x_in])) * may no longer be need, relaced with motivator 
							motivator; 

				if (maxtransy[y_in] < trans[x_in][y_in]) maxtransy[y_in] = trans[x_in][y_in];
				if (maxtransx[x_in] < trans[x_in][y_in]) maxtransx[x_in] = trans[x_in][y_in];
				if (mintransy[y_in] > trans[x_in][y_in]) mintransy[y_in] = trans[x_in][y_in];
				if (mintransx[x_in] > trans[x_in][y_in]) mintransx[x_in] = trans[x_in][y_in];
				
				avetransx[x_in] += trans[x_in][y_in];
				avetransy[y_in] += trans[x_in][y_in];
			}
		}

		for (int x_in = 0; x_in < layers[x_lay].size(); ++x_in) {
			avetransx[x_in] = avetransx[x_in] / layers[x_lay].size() - mintransx[x_in];
			maxtransx[x_in] = maxtransx[x_in] - mintransx[x_in];
		} for (int y_in = 0; y_in < layers[linking[x_lay][y_lay]].size(); ++y_in) {
			avetransy[y_in] = avetransy[y_in] / layers[linking[x_lay][y_lay]].size() - mintransy[y_in];
			maxtransy[y_in] = maxtransy[y_in] - mintransy[y_in];
		}
		
		// changing weights  ------------------------------------------------->
		if (is_training) {
			for (int y_in = 0; y_in < layers[linking[x_lay][y_lay]].size(); ++y_in) {
				for (int x_in = 0; x_in < layers[x_lay].size(); ++x_in) {
					// ----------------------------------------------
					// w = w + (tw - w) * m
					target = ((trans[x_in][y_in] - mintransy[y_in]) / maxtransy[y_in] + 
								(trans[x_in][y_in] - mintransx[x_in]) / maxtransx[x_in]) / 2;
					weights[x_lay][y_lay][x_in][y_in] += (target - weights[x_lay][y_lay][x_in][y_in]) * abs(motivator);
					// ----------------------------------------------

					// if there will be some troubles with double operations
					if (weights[x_lay][y_lay][x_in][y_in] < 0 || isnan(weights[x_lay][y_lay][x_in][y_in]))
						weights[x_lay][y_lay][x_in][y_in] = 0;
				}
			}
		}


		// pseudo activation function ----------------------------------------->
		
		for (int x_in = 0; x_in < layers[x_lay].size(); ++x_in) 
			layers[x_lay][x_in] *= avetransx[x_in] / maxtransx[x_in];
			
		for (int y_in = 0; y_in < layers[linking[x_lay][y_lay]].size(); ++y_in)
			layers[linking[x_lay][y_lay]][y_in] *= avetransy[y_in] / maxtransy[y_in];
		
		// -------------------------------------------------------------------->
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

		// -------------------------------------------------------------------->
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
