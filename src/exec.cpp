// EY = accumulator of y values on each layer
// EX = accumulator of x values on each layer

// F = (y_j / EY) * weght_ij 
// P = (1 - y_j / (y_j + peak)) / (EX / x_i)
// Q = distance between neurons
// Q, = average distance between neurons

// execution iteration
void kernel::model::exec (bool is_training, double motivator)
{
    for (uint32_t x_lay = 0; x_lay < layers.size(); ++x_lay) { // get x layers
        for (uint32_t y_lay = 0; y_lay < linking[x_lay].size(); ++y_lay) { // get y layers
            
			buf_d[0] = 0; // sum of an y's vals (in y layer) as EY
            for (uint32_t y_in = 0; y_in < layers[linking[x_lay][y_lay]].size(); ++y_in) {
                buf_d[0] += layers[linking[x_lay][y_lay]][y_in];
            }

            buf_d[3] = 0; // sum of an x's vals (in x layer) as EX
            for (uint32_t x_in = 0; x_in < layers[x_lay].size(); ++x_in) {
                buf_d[3] += layers[x_lay][x_in];
            }

            buf_mat = arma::Mat<double>(layers[x_lay].size(), layers[linking[x_lay][y_lay]].size());

            for (uint32_t x_in = 0; x_in < layers[x_lay].size(); ++x_in) {

				// collecting accumulation of potentials as EF
				buf_d[2] = 0; // sum of potentials
                for (uint32_t y_in = 0; y_in < layers[linking[x_lay][y_lay]].size(); ++y_in) {
                    buf_d[2] += layers[linking[x_lay][y_lay]][y_in] / buf_d[0] *
                                weights[x_lay][y_lay].at(x_in, y_in);
                }

                // collecting transmitted signal ----------------------------->
                for (uint32_t y_in = 0; y_in < layers[linking[x_lay][y_lay]].size(); ++y_in) {
                    buf_mat.at(x_in, y_in) =
                    /*   F   */((layers[linking[x_lay][y_lay]][y_in] / buf_d[0] * weights[x_lay][y_lay].at(x_in, y_in)) / buf_d[2]) *
                    /*   P   */(((1 - layers[linking[x_lay][y_lay]][y_in] / (layers[linking[x_lay][y_lay]][y_in] + neupick)) / (buf_d[3] / layers[x_lay][x_in])) +
                    /* Q / Q,*/(conducts[x_lay][y_lay].at(x_in, y_in) / ((double(2) / std::max(layers[linking[x_lay][y_lay]].size(), layers[x_lay].size()))))) / 2;
                }
            }

            buffer_vec = std::vector<double>(layers[linking[x_lay][y_lay]].size(), 0);

            // transform F and P in end result --------------------------------->
            for (uint32_t x_in = 0; x_in < layers[x_lay].size(); ++x_in) {
                for (uint32_t y_in = 0; y_in < layers[linking[x_lay][y_lay]].size(); ++y_in) {
                    buf_mat.at(x_in, y_in) *= layers[x_lay][x_in]; // N * ((F * P) * (Q / Q,)) -> result
                    buffer_vec[y_in] += buf_mat.at(x_in, y_in);
                }
            }

			double bufdb;
            // changing weights  ------------------------------------------------->
            if (is_training) {
                for (int y_in = 0; y_in < layers[linking[x_lay][y_lay]].size(); ++y_in) {
                    for (int x_in = 0; x_in < layers[x_lay].size(); ++x_in) {
						// you can change it, here are you will chose next theoretical weight power
						bufdb = buf_mat.at(x_in, y_in) / buffer_vec[y_in];

                        if (bufdb > weights[x_lay][y_lay].at(x_in, y_in))
							weights[x_lay][y_lay].at(x_in, y_in) += (bufdb - weights[x_lay][y_lay].at(x_in, y_in)) * motivator;
						else 
							weights[x_lay][y_lay].at(x_in, y_in) -= (weights[x_lay][y_lay].at(x_in, y_in) - bufdb) * motivator;
						// ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

						// if weight will become in very small value
						if (weights[x_lay][y_lay].at(x_in, y_in) < 0 || std::isnan(weights[x_lay][y_lay].at(x_in, y_in)))
                            weights[x_lay][y_lay].at(x_in, y_in) = 0;
                    }
                }
			}

            // transform values of x layers and y layers -------------------------->
            for (uint32_t x_in = 0; x_in < layers[x_lay].size(); ++x_in) 
                for (uint32_t y_in = 0; y_in < layers[linking[x_lay][y_lay]].size(); ++y_in) {
                    layers[x_lay][x_in] -= buf_mat.at(x_in, y_in);
                    layers[linking[x_lay][y_lay]][y_in] += buf_mat.at(x_in, y_in);
                }

            // normalize y layer ------------------------------------------------>
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
            
            // normalize x layer------------------------------------------------->
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
