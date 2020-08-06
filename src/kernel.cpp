using namespace std;

// initialization
kernel::model::model ( vector<pair<unsigned int, vector<unsigned int>>> init_arch, 
	double init_defval, double init_cover, double init_peak, 
	double init_impulse, string init_name ) 
{
	int neusy;

	defval = init_defval;
	name = init_name;
	impulse = init_impulse;
	cover = init_cover;
	neupeak = init_peak;

	// initialization of layers and linked layers
	for (uint32_t it = 0; it < init_arch.size(); ++it) {
		layers.push_back(vector<double>(init_arch[it].first, defval));
		linking.push_back(init_arch[it].second);
		weights.push_back(vector<vector<vector<double>>>{});
		conducts.push_back(vector<vector<vector<double>>>{});
	}

	// initialization of all connections between neurons
	for (int it = 0; it < init_arch.size(); ++it) {
		for (int it_1 = 0; it_1 < linking[it].size(); ++it_1) {

			weights[it].push_back(vector<vector<double>>(layers[it].size(), (vector<double>(layers[linking[it][it_1]].size(), 0))));
			conducts[it].push_back(vector<vector<double>>(layers[it].size(), (vector<double>(layers[linking[it][it_1]].size(), 0))));

			// initialization of distances
			if (layers[it].size() <= layers[linking[it][it_1]].size()) {
				neusy = layers[linking[it][it_1]].size() * cover / 2;

				for (int x = 0; x < layers[it].size(); ++x)
					for (int y = 0; y < layers[linking[it][it_1]].size(); ++y) {
						if (y > double(x) / layers[it].size() * layers[linking[it][it_1]].size() - neusy &&
							y < double(x) / layers[it].size() * layers[linking[it][it_1]].size() + neusy)
								conducts[it][it_1][x][y] = 1;
						else 	
								conducts[it][it_1][x][y] = 0;
					}
			} else {
				neusy = layers[it].size() * cover / 2;

				for (int x = 0; x < layers[linking[it][it_1]].size(); ++x)
					for (int y = 0; y < layers[it].size(); ++y) {
						if (y > double(x) / layers[linking[it][it_1]].size() * layers[it].size() - neusy &&
							y < double(x) / layers[linking[it][it_1]].size() * layers[it].size() + neusy)
								conducts[it][it_1][y][x] = 1;
						else 	
								conducts[it][it_1][y][x] = 0;
					}

			} 

		}
	}

}

// setting up input values
void kernel::model::setIn (vector<double> blank, int lay, int base) 
{
	for (int it = 0; it < blank.size(); ++it) 
		layers[lay][it + base] = blank[it];
}

// getting of the out layer
vector<double> kernel::model::getOut (void) 
{
	return layers[layers.size() - 1];
}

// throwing out of the values of the latter layer
void kernel::model::dropOut (void) 
{
	for (int it = 0; it < layers[layers.size() - 1].size(); ++it) 
		layers[layers.size() - 1][it] = defval;
}

// throwing out of network neurons
void kernel::model::dropPart (double part) 
{
	double sum;
	for (int x_lay = 0; x_lay < layers.size(); ++x_lay) {
		sum = 0;

		for (int x_in = 0; x_in < layers[x_lay].size(); ++x_in) 
			sum += layers[x_lay][x_in];

		sum /= layers[x_lay].size();

		for (int x_in = 0; x_in < layers[x_lay].size(); ++x_in) {
			if (layers[x_lay][x_in] > sum)
				layers[x_lay][x_in] -= (layers[x_lay][x_in] - defval) * part;

			if (layers[x_lay][x_in] < defval) 
				layers[x_lay][x_in] = defval;
		}
	}
}
