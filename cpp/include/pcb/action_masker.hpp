// ActionMasker: computes the set of physically legal moves for the routing
// head, in continuous space, against a Bounding Volume Hierarchy (R-tree).
//
// Contract with the Python agent (python/pcb_router/model.py):
//   - kNumAngleBins / kMaxLayers / kNumActionTypes must match the constants
//     in model.py. The mask arrays here are flattened directly into the
//     tensors the actor multiplies its logits against.
//   - For EXTEND actions the agent emits a *fraction* in (0,1); the engine
//     scales it by ActionMask::max_distance[bin], so any sampled EXTEND is
//     legal by construction (no post-hoc DRC rejection).
//
// Units: millimetres, double precision. Layer 0 = top copper.

#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

#include <boost/geometry.hpp>
#include <boost/geometry/index/rtree.hpp>

namespace pcb {

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

using Point2D  = bg::model::d2::point_xy<double>;
using Segment  = bg::model::segment<Point2D>;
using Polygon  = bg::model::polygon<Point2D>;
using Box      = bg::model::box<Point2D>;

// ---------------------------------------------------------------- geometry --

enum class ObstacleKind : std::uint8_t { Trace, Pad, Via, KeepOut, BoardEdge };

struct Obstacle {
  Polygon shape;          // copper/keep-out footprint on this layer
  int layer_lo = 0;       // vias span [layer_lo, layer_hi]; planar items have
  int layer_hi = 0;       //   layer_lo == layer_hi
  int net_id = -1;        // -1: no net (keep-outs, board edge)
  ObstacleKind kind = ObstacleKind::Trace;
};

// R-tree over obstacle AABBs; payload indexes into the obstacle store.
using RTreeValue = std::pair<Box, std::uint32_t>;
using RTree = bgi::rtree<RTreeValue, bgi::rstar<16>>;

// ------------------------------------------------------------ design rules --

struct DesignRules {
  double trace_width = 0.15;        // default routed width
  double trace_clearance = 0.15;    // copper-to-copper, different nets
  double via_drill = 0.30;
  double via_annular = 0.15;        // pad radius = drill/2 + annular
  double via_clearance = 0.20;
  double min_segment_length = 0.05; // reject degenerate extensions
  double board_clearance = 0.25;    // copper to board edge
};

// -------------------------------------------------------------- action mask --

inline constexpr int kNumActionTypes = 3;   // EXTEND, PLACE_VIA, COMMIT_NET
inline constexpr int kNumAngleBins  = 64;   // 5.625 deg per bin
inline constexpr int kMaxLayers     = 12;

enum class ActionType : std::uint8_t { Extend = 0, PlaceVia = 1, CommitNet = 2 };

struct ActionMask {
  std::array<std::uint8_t, kNumActionTypes> type_mask{};   // 1 = legal
  std::array<std::uint8_t, kNumAngleBins>  angle_mask{};   // per-direction
  std::array<double,       kNumAngleBins>  max_distance{}; // mm legal per bin
  std::array<std::uint8_t, kMaxLayers>     layer_mask{};   // legal via targets
};

// Current state of the net being routed.
struct RoutingHead {
  Point2D position;
  int layer = 0;
  int net_id = -1;
  double trace_width = 0.15;   // may differ per net class (e.g. power)
  Point2D target;              // pad the head must reach
};

// --------------------------------------------------------------- the masker --

class ActionMasker {
 public:
  ActionMasker(const RTree& index,
               const std::vector<Obstacle>& obstacles,
               const DesignRules& rules,
               int num_layers);

  // Full mask for the agent. Cost target: < 50 us per call. Implementation
  // strategy: ONE windowed R-tree query around the head (radius = lookahead),
  // then clip all kNumAngleBins rays against the handful of local obstacles
  // in-cache, instead of kNumAngleBins independent tree queries.
  ActionMask ComputeMask(const RoutingHead& head,
                         double lookahead_mm = 5.0) const;

  // Farthest legal extension from `origin` along `angle_rad` on `layer`,
  // modelled as a capsule sweep of radius trace_width/2, expanded by the
  // applicable clearance (Minkowski). Same-net copper is not an obstacle.
  // Returns 0.0 if even min_segment_length collides.
  double MaxLegalDistance(const Point2D& origin, double angle_rad,
                          int layer, int net_id, double trace_width,
                          double max_range) const;

  // True iff the swept trace segment violates no clearance rule.
  bool IsSegmentLegal(const Segment& seg, int layer, int net_id,
                      double trace_width) const;

  // True iff a via (drill + annular + clearance) fits at `pos` spanning
  // [from_layer, to_layer] without shorting foreign nets on ANY layer crossed.
  bool CanPlaceVia(const Point2D& pos, int from_layer, int to_layer,
                   int net_id) const;

  // True iff the head is within snap distance of its target pad and the
  // closing segment is legal (enables COMMIT_NET in the type mask).
  bool CanCommit(const RoutingHead& head, double snap_mm = 0.10) const;

 private:
  // Windowed candidate fetch: all obstacles whose AABB intersects `window`
  // and whose layer span covers `layer`, excluding same-net copper.
  void QueryWindow(const Box& window, int layer, int net_id,
                   std::vector<std::uint32_t>* out) const;

  const RTree& index_;
  const std::vector<Obstacle>& obstacles_;
  DesignRules rules_;
  int num_layers_;
};

}  // namespace pcb
