// initialization / overload - 0
kernel::model::model (std::vector<std::pair<uint32_t, std::vector<uint32_t>>> init_arch = {}, double init_default_v = 1, double init_reserve = 1, double maxval = 1, double impulse_k = 1, std::string init_name = "new_kernel") 
{
	default_v = init_default_v;
	name = init_name;
	impulse = impulse_k;
	modify = init_reserve;
	neupeak = maxval;

	// init layers and links
	for (uint32_t it = 0; it < init_arch.size(); ++it) {
		layers.push_back(std::vector<double>(init_arch[it].first, default_v));
		linking.push_back(init_arch[it].second);
		weights.push_back(std::vector<arma::Mat<double>>{});
		conducts.push_back(std::vector<arma::Mat<double>>{});
	}

	// init all parts of connection between neurons, as distances and weights
	for (uint32_t it = 0; it < init_arch.size(); ++it) {
		for (uint32_t it_1 = 0; it_1 < linking[it].size(); ++it_1) {

			weights[it].push_back(arma::Mat<double>(layers[it].size(), layers[linking[it][it_1]].size()).ones());
			weights[it][weights[it].size() - 1] /= (layers[it].size() * layers[linking[it][it_1]].size());
			int neusy;

			conducts[it].push_back(arma::Mat<double>(layers[it].size(), layers[linking[it][it_1]].size()));

			// init of distances
			if (layers[it].size() <= layers[linking[it][it_1]].size()) {
				neusy = layers[linking[it][it_1]].size() * modify / 2;

				for (int x = 0; x < layers[it].size(); ++x)
					for (int y = 0; y < layers[linking[it][it_1]].size(); ++y) {
						if (y > double(x) / layers[it].size() * layers[linking[it][it_1]].size() - neusy &&
							y < double(x) / layers[it].size() * layers[linking[it][it_1]].size() + neusy)
							conducts[it][it_1].at(x, y) = 1;
						else 	
							conducts[it][it_1].at(x, y) = 0;
					}
			} else {
				neusy = layers[it].size() * modify / 2;

				for (int x = 0; x < layers[linking[it][it_1]].size(); ++x)
					for (int y = 0; y < layers[it].size(); ++y) {
						if (y > double(x) / layers[linking[it][it_1]].size() * layers[it].size() - neusy &&
							y < double(x) / layers[linking[it][it_1]].size() * layers[it].size() + neusy)
							conducts[it][it_1].at(y, x) = 1;
						else 	
							conducts[it][it_1].at(y, x) = 0;
					}

			} 

		}
	}
}

// initialization / overload - 1 ~ from file stream
kernel::model::model (std::string path) 
{
	open(path);
}

// serialize model
void kernel::model::backup(std::string path) 
{
	std::ofstream log = std::ofstream(path + name + ".kr");

	log << name << " ";
	log << default_v << " ";
	log << layers.size() << " ";
	log << modify << " ";
	log << neupeak << " ";
	log << impulse << " ";

	for (uint32_t it = 0; it < layers.size(); ++it) {
		log << layers[it].size() << " ";

		for (uint32_t it_1 = 0; it_1 < layers[it].size(); ++it_1) {
			log << layers[it][it_1] << " ";
		}
	}

	log << linking.size() << " ";
	for (uint32_t it = 0; it < linking.size(); ++it) {
		log << linking[it].size() << " ";

		for (uint32_t it_1 = 0; it_1 < linking[it].size(); ++it_1) {
			log << linking[it][it_1] << " ";
		}
	}

	log << weights.size() << " ";
	for (uint32_t it = 0; it < weights.size(); ++it) {
		log << weights[it].size() << " ";

		for (uint32_t it_1 = 0; it_1 < weights[it].size(); ++it_1) {
			log << weights[it][it_1].n_rows << " " << weights[it][it_1].n_cols << " ";

			for (uint32_t y = 0; y < weights[it][it_1].n_rows; ++y) {
				for (uint32_t x = 0; x < weights[it][it_1].n_cols; ++x) {
					log << weights[it][it_1].at(y, x) << " ";
				}
			}
		}
	}

	log << conducts.size() << " ";
	for (uint32_t it = 0; it < conducts.size(); ++it) {
		log << conducts[it].size() << " ";

		for (uint32_t it_1 = 0; it_1 < conducts[it].size(); ++it_1) {
			log << conducts[it][it_1].n_rows << " " << conducts[it][it_1].n_cols << " ";

			for (uint32_t y = 0; y < conducts[it][it_1].n_rows; ++y) {
				for (uint32_t x = 0; x < conducts[it][it_1].n_cols; ++x) {
					log << conducts[it][it_1].at(y, x) << " ";
				}
			}
		}
	}

	log.close();
}

// serialize kernel
void kernel::model::open(std::string path) 
{
	std::ifstream log = std::ifstream(path);

	log >> name;
	log >> default_v;
	log >> buf_ui[0];
	log >> modify;
	log >> neupeak;
	log >> impulse;

	for (uint32_t it = 0; it < buf_ui[0]; ++it) {
		layers.push_back({});
		log >> buf_ui[1];

		for (uint32_t it_1 = 0; it_1 < buf_ui[1]; ++it_1) {
			log >> buffer_T;
			layers[it].push_back(buffer_T);
		}
	}

	log >> buf_ui[0];
	for (uint32_t it = 0; it < buf_ui[0]; ++it) {
		linking.push_back({});
		log >> buf_ui[1];

		for (uint32_t it_1 = 0; it_1 < buf_ui[1]; ++it_1) {
			log >> buf_ui[2];
			linking[it].push_back(buf_ui[2]);
		}
	}

	log >> buf_ui[0];
	for (uint32_t it = 0; it < buf_ui[0]; ++it) {
		weights.push_back(std::vector<arma::Mat<double>>{});
		log >> buf_ui[1];

		for (uint32_t it_1 = 0; it_1 < buf_ui[1]; ++it_1) {
			log >> buf_ui[2] >> buf_ui[3];
			weights[it].push_back(arma::Mat<double>(buf_ui[2], buf_ui[3]));

			for (uint32_t y = 0; y < buf_ui[2]; ++y) {
				for (uint32_t x = 0; x < buf_ui[3]; ++x) {
					log >> weights[it][it_1].at(y, x);
				}
			}
		}
	}

	log >> buf_ui[0];
	for (uint32_t it = 0; it < buf_ui[0]; ++it) {
		conducts.push_back(std::vector<arma::Mat<double>>{});
		log >> buf_ui[1];

		for (uint32_t it_1 = 0; it_1 < buf_ui[1]; ++it_1) {
			log >> buf_ui[2] >> buf_ui[3];
			conducts[it].push_back(arma::Mat<double>(buf_ui[2], buf_ui[3]));

			for (uint32_t y = 0; y < buf_ui[2]; ++y) {
				for (uint32_t x = 0; x < buf_ui[3]; ++x) {
					log >> conducts[it][it_1].at(y, x);
				}
			}
		}
	}

	log.close();
}

// drop out buffers
void kernel::model::dropBuffers (void) 
{
	buffer_vec.clear();
	buffer_T = 0;

	buf_ui[0] = 0; buf_ui[1] = 0;
	buf_ui[2] = 0; buf_ui[3] = 0;

	buf_d[0] = 0; buf_d[1] = 0;
	buf_d[2] = 0; buf_d[3] = 0;
}

// set up input values
void kernel::model::setIn (std::vector<double> blank, int lay, int base) 
{
	for (uint32_t it = 0; it < blank.size(); ++it) layers[lay][it + base] = blank[it];
}

// get out as vector
std::vector<double> kernel::model::getOut (void) 
{
	return layers[layers.size() - 1];
}

// dropping out values of an out
void kernel::model::dropOut (void) 
{
	for (uint32_t it = 0; it < layers[layers.size() - 1].size(); ++it) layers[layers.size() - 1][it] = default_v;
}

// dropping out a part of each neuron layer
void kernel::model::dropPart (double part) 
{
    for (uint32_t x_lay = 0; x_lay < layers.size(); ++x_lay) {
		double sum = 0;
		for (uint32_t x_in = 0; x_in < layers[x_lay].size(); ++x_in) 
			sum += layers[x_lay][x_in];
		sum /= layers[x_lay].size();

		for (uint32_t x_in = 0; x_in < layers[x_lay].size(); ++x_in) {
			if (layers[x_lay][x_in] > sum)
				layers[x_lay][x_in] -= (layers[x_lay][x_in] - default_v) * part;

			if (layers[x_lay][x_in] < default_v) layers[x_lay][x_in] = default_v;
		}
	}
}
