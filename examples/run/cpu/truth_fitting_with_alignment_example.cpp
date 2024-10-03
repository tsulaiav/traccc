/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2023-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Project include(s).
#include "traccc/definitions/common.hpp"
#include "traccc/definitions/primitives.hpp"
#include "traccc/fitting/fitting_algorithm.hpp"
#include "traccc/fitting/kalman_filter/kalman_fitter.hpp"
#include "traccc/io/read_geometry.hpp"
#include "traccc/io/read_measurements.hpp"
#include "traccc/io/utils.hpp"
#include "traccc/options/detector.hpp"
#include "traccc/options/input_data.hpp"
#include "traccc/options/performance.hpp"
#include "traccc/options/program_options.hpp"
#include "traccc/options/track_propagation.hpp"
#include "traccc/resolution/fitting_performance_writer.hpp"
#include "traccc/utils/seed_generator.hpp"

// Detray include(s).
#include "detray/core/detector.hpp"
#include "detray/core/detector_metadata.hpp"
#include "detray/detectors/bfield.hpp"
#include "detray/io/frontend/detector_reader.hpp"
#include "detray/navigation/navigator.hpp"
#include "detray/propagator/propagator.hpp"
#include "detray/propagator/rk_stepper.hpp"

// VecMem include(s).
#include <vecmem/memory/host_memory_resource.hpp>

// System include(s).
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>

using namespace traccc;
namespace po = boost::program_options;

// The main routine
//
int main(int argc, char* argv[]) {

    // Program options.
    traccc::opts::detector detector_opts;
    traccc::opts::input_data input_opts;
    traccc::opts::track_propagation propagation_opts;
    traccc::opts::performance performance_opts;
    traccc::opts::program_options program_opts{
        "Truth Track Fitting on the Host",
        {detector_opts, input_opts, propagation_opts, performance_opts},
        argc,
        argv};

    /// Type declarations
    using host_detector_type = detray::detector<detray::default_metadata,
                                                detray::host_container_types>;

    using b_field_t = covfie::field<detray::bfield::const_bknd_t>;
    using rk_stepper_type =
        detray::rk_stepper<b_field_t::view_t, traccc::default_algebra,
                           detray::constrained_step<>>;

    using host_navigator_type = detray::navigator<const host_detector_type>;
    using host_fitter_type =
        traccc::kalman_fitter<rk_stepper_type, host_navigator_type>;

    // Memory resources used by the application.
    vecmem::host_memory_resource host_mr;

    // Performance writer
    traccc::fitting_performance_writer fit_performance_writer(
        traccc::fitting_performance_writer::config{});

    /*****************************
     * Build a geometry
     *****************************/

    // B field value and its type
    // @TODO: Set B field as argument
    const traccc::vector3 B{0, 0, 2 * detray::unit<traccc::scalar>::T};
    auto field = detray::bfield::create_const_field(B);

    // Read the detector
    detray::io::detector_reader_config reader_cfg{};
    reader_cfg.add_file(traccc::io::data_directory() +
                        detector_opts.detector_file);
    if (!detector_opts.material_file.empty()) {
        reader_cfg.add_file(traccc::io::data_directory() +
                            detector_opts.material_file);
    }
    if (!detector_opts.grid_file.empty()) {
        reader_cfg.add_file(traccc::io::data_directory() +
                            detector_opts.grid_file);
    }
    const auto [host_det, names] =
        detray::io::read_detector<host_detector_type>(host_mr, reader_cfg);

    using transform_store = host_detector_type::transform_container;
    using transform_vector = transform_store::base_type;
    
    const transform_store& transforms = host_det.transform_store();
    transform_vector newtransforms;
    newtransforms.reserve(transforms.size());
    for (const auto& transform : transforms) {
      newtransforms.push_back(transform);
    }

    /*
    // Ugly business ...
    const transform_store* ptr_transforms_const = &transforms;
    transform_store* ptr_transforms = const_cast<transform_store*>(ptr_transforms_const);
    // ... Ugly business

    ptr_transforms->fix_context_size();
    ptr_transforms->dummy();
    ptr_transforms->dump_info();
    ptr_transforms->add_context(std::move(newtransforms));
    ptr_transforms->dump_info();
    */
    
    /*****************************
     * Do the reconstruction
     *****************************/

    /// Standard deviations for seed track parameters
    static constexpr std::array<scalar, e_bound_size> stddevs = {
        0.03f * detray::unit<scalar>::mm,
        0.03f * detray::unit<scalar>::mm,
        0.017f,
        0.017f,
        0.01f / detray::unit<scalar>::GeV,
        1.f * detray::unit<scalar>::ns};

    // Fitting algorithm objects
    typename traccc::fitting_algorithm<host_fitter_type>::config_type fit_cfg0;
    fit_cfg0.propagation = propagation_opts;
    fit_cfg0.propagation.context = detray::geometry_context{0};
    traccc::fitting_algorithm<host_fitter_type> host_fitting0(fit_cfg0);

    typename traccc::fitting_algorithm<host_fitter_type>::config_type fit_cfg1;
    fit_cfg1.propagation = propagation_opts;
    fit_cfg1.propagation.context = detray::geometry_context{1};
    traccc::fitting_algorithm<host_fitter_type> host_fitting1(fit_cfg1);

    // Seed generators
    traccc::seed_generator<host_detector_type> sg0(host_det, stddevs, 0, fit_cfg0.propagation.context);
    traccc::seed_generator<host_detector_type> sg1(host_det, stddevs, 0, fit_cfg1.propagation.context);
    
    // Iterate over events
    for (unsigned int event = input_opts.skip;
         event < input_opts.events + input_opts.skip; ++event) {

        // Truth Track Candidates
        traccc::event_data evt_data(input_opts.directory, event, host_mr,
				    input_opts.use_acts_geom_source, &host_det,
				    input_opts.format, false);

	// For the first half of events
	if ((event - input_opts.skip) / (input_opts.events / 2) == 0) {
	  traccc::track_candidate_container_types::host truth_track_candidates =
            evt_data.generate_truth_candidates(sg0, host_mr);
	  
	  // Run fitting
	  auto track_states =
	    host_fitting0(host_det, field, truth_track_candidates);

	  unsigned int n_fitted_tracks = track_states.size();
	  std::cout << "Number of fitted tracks (1st alignment): "
		    << n_fitted_tracks << std::endl;
	}
	// For the second half of events
	else {
	  // Change the alignment setting
	  traccc::track_candidate_container_types::host truth_track_candidates =
            evt_data.generate_truth_candidates(sg1, host_mr);

	  // Run fitting
	  auto track_states =
	    host_fitting1(host_det, field, truth_track_candidates);

	  unsigned int n_fitted_tracks = track_states.size();
	  std::cout << "Number of fitted tracks (2nd alignment): "
		    << n_fitted_tracks << std::endl;	  
	}
    }
    return EXIT_SUCCESS;
}
