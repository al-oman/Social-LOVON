#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
#  Social-LOVON headless evaluation sweep
#
#  Iterates over:
#    - socialnav enabled / disabled
#    - robot_theta
#    - human_num
#    - human v_pref (speed)
#    - human policy (orca / linear)
#
#  Each combination gets a temporary env config with the parameter
#  overrides, then runs deploy_headless.py --headless.
#  Results CSVs are collected into evaluation/results/<timestamp>/
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Paths ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEPLOY="$PROJECT_ROOT/deploy/deploy_headless.py"
BASE_ENV_CONFIG="$PROJECT_ROOT/configs/env_lovon.config"
POLICY_CONFIG="$PROJECT_ROOT/configs/policy_lovon.config"

# Model paths are relative to project root, so we must run from there
cd "$PROJECT_ROOT"

# ── Sweep parameters (edit these) ──
SOCIALNAV_FLAGS=("" "--socialnav_enabled")          # disabled / enabled
ROBOT_THETAS=(1.5708 0.7854 3.1416)                 # pi/2, pi/4, pi  (radians)
HUMAN_NUMS=(1 2 3 5)
HUMAN_SPEEDS=(0.5 1.0 1.5)                          # v_pref (m/s)
HUMAN_POLICIES=("orca" "linear")

# ── Per-run settings ──
NUM_EPISODES=10
MAX_STEPS=500
MISSION="move to the handbag at speed of 0.5 m/s"

# ── Output directory ──
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="$SCRIPT_DIR/results/$TIMESTAMP"
mkdir -p "$RESULTS_DIR"

# ── Helper: generate a temporary env config with overrides ──
make_env_config() {
    local human_num="$1"
    local human_speed="$2"
    local human_policy="$3"
    local tmp_config
    tmp_config=$(mktemp /tmp/env_lovon_XXXXXX.config)

    # Copy base config and override the three fields
    sed \
        -e "s/^human_num *=.*/human_num = ${human_num}/" \
        -e "s/^v_pref *=.*/v_pref = ${human_speed}/" \
        -e "s/^policy *=.*/policy = ${human_policy}/" \
        "$BASE_ENV_CONFIG" > "$tmp_config"

    echo "$tmp_config"
}

# ── Sweep ──
RUN=0
TOTAL=$(( ${#SOCIALNAV_FLAGS[@]} * ${#ROBOT_THETAS[@]} * ${#HUMAN_NUMS[@]} * ${#HUMAN_SPEEDS[@]} * ${#HUMAN_POLICIES[@]} ))

echo "============================================================"
echo "  Social-LOVON evaluation sweep"
echo "  ${TOTAL} configurations x ${NUM_EPISODES} episodes each"
echo "  Results: ${RESULTS_DIR}"
echo "============================================================"
echo ""

for SOCIALNAV in "${SOCIALNAV_FLAGS[@]}"; do
    for THETA in "${ROBOT_THETAS[@]}"; do
        for NHUMANS in "${HUMAN_NUMS[@]}"; do
            for SPEED in "${HUMAN_SPEEDS[@]}"; do
                for HPOLICY in "${HUMAN_POLICIES[@]}"; do

                    RUN=$((RUN + 1))
                    SN_LABEL=$( [[ -n "$SOCIALNAV" ]] && echo "on" || echo "off" )
                    TAG="sn${SN_LABEL}_theta${THETA}_h${NHUMANS}_spd${SPEED}_${HPOLICY}"

                    echo "────────────────────────────────────────────────────"
                    echo "  [${RUN}/${TOTAL}]  ${TAG}"
                    echo "────────────────────────────────────────────────────"

                    # Generate temp config
                    TMP_CONFIG=$(make_env_config "$NHUMANS" "$SPEED" "$HPOLICY")

                    # Run headless evaluation
                    python "$DEPLOY" \
                        --headless \
                        --num_episodes "$NUM_EPISODES" \
                        --max_steps "$MAX_STEPS" \
                        --mission_instruction "$MISSION" \
                        --robot_theta "$THETA" \
                        --env_config "$TMP_CONFIG" \
                        --policy_config "$POLICY_CONFIG" \
                        $SOCIALNAV

                    # Move the generated CSV into results dir with descriptive name
                    # deploy_headless.py writes eval_results_<ts>.csv in cwd
                    LATEST_CSV=$(ls -t eval_results_*.csv 2>/dev/null | head -1)
                    if [[ -n "$LATEST_CSV" ]]; then
                        mv "$LATEST_CSV" "$RESULTS_DIR/${TAG}.csv"
                    fi

                    # Clean up temp config
                    rm -f "$TMP_CONFIG"

                    echo ""
                done
            done
        done
    done
done

echo "============================================================"
echo "  Sweep complete.  ${TOTAL} runs finished."
echo "  Results saved to: ${RESULTS_DIR}"
echo "============================================================"
