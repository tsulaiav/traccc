/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022-2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

// Local include(s).
#include "full_chain_algorithm.hpp"

namespace traccc {

full_chain_algorithm::full_chain_algorithm(
    vecmem::memory_resource& mr, const clustering_algorithm::config_type&,
    const seedfinder_config& finder_config,
    const spacepoint_grid_config& grid_config,
    const seedfilter_config& filter_config,
    const finding_algorithm::config_type& finding_config,
    const fitting_algorithm::config_type& fitting_config,
    const silicon_detector_description::host& det_descr,
    detector_type* detector)
    : m_field_vec{0.f, 0.f, finder_config.bFieldInZ},
      m_field(detray::bfield::create_const_field(m_field_vec)),
      m_det_descr(det_descr),
      m_detector(detector),
      m_clusterization(mr),
      m_spacepoint_formation(mr),
      m_seeding(finder_config, grid_config, filter_config, mr),
      m_track_parameter_estimation(mr),
      m_finding(finding_config),
      m_fitting(fitting_config),
      m_finder_config(finder_config),
      m_grid_config(grid_config),
      m_filter_config(filter_config),
      m_finding_config(finding_config),
      m_fitting_config(fitting_config) {}

full_chain_algorithm::output_type full_chain_algorithm::operator()(
    const cell_collection_types::host& cells) const {

    // Create a data object for the detector description.
    const silicon_detector_description::const_data det_descr_data =
        vecmem::get_data(m_det_descr.get());

    // Run the clusterization.
    const host::clusterization_algorithm::output_type measurements =
        m_clusterization(vecmem::get_data(cells), det_descr_data);

    // If we have a Detray detector, run the seeding track finding and fitting.
    if (m_detector != nullptr) {

        // Run the seed-finding.
        const host::spacepoint_formation_algorithm<
            const detector_type>::output_type spacepoints =
            m_spacepoint_formation(*m_detector, vecmem::get_data(measurements));
        const track_params_estimation::output_type track_params =
            m_track_parameter_estimation(spacepoints, m_seeding(spacepoints),
                                         m_field_vec);

        // Return the final container, after track finding and fitting.
        return m_fitting(
            *m_detector, m_field,
            m_finding(*m_detector, m_field, measurements, track_params));

    }
    // If not, just return an empty object.
    else {

        // Return an empty object.
        return {};
    }
}

}  // namespace traccc
