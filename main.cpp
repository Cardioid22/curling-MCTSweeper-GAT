#include <cassert>
#include <cstdlib>
#include <stdexcept>
#include <iostream>
#include <thread>
#include <boost/asio.hpp>
#include "digitalcurling3/digitalcurling3.hpp"

#include <boost/unordered_map.hpp>


namespace dc = digitalcurling3;

namespace {

/// \brief GameState::Stones のインデックス．
struct StoneIndex
{ 
    size_t team;
    size_t stone;
};

// ハッシュ関数
struct pair_hash {
    std::size_t operator()(const std::pair<float, float>& p) const {
        auto h1 = std::hash<float>{}(p.first);
        auto h2 = std::hash<float>{}(p.second);
        return h1 ^ (h2 << 1); // ビットシフトでハッシュを合成
    }
};

// ショットシミュレーション用変数
dc::Team g_team; /// 自分のチームID
dc::GameSetting g_game_setting; // ゲーム設定
std::unique_ptr<dc::ISimulator> g_simulator; // シミュレーターのインターフェイス
std::unique_ptr<dc::ISimulatorStorage> g_simulator_storage; // シミュレーションの状態を保存するストレージ
std::array<std::unique_ptr<dc::IPlayer>, 4> g_players; // ゲームプレイヤー

// グリッドサイズ
constexpr float kHouseRadius = 1.829f; // ハウスの半径
constexpr float kPlayAreaXMin = -2.375f;
constexpr float kPlayAreaXMax = 2.375f;
constexpr float kHouseCenterY = 38.405f;
constexpr float kBackLine = 40.234f;
constexpr double PI = 3.141592653589793;
dc::Vector2 kCenter(0.0f, kHouseCenterY);

// MCTS用のノード
struct Node {
    dc::GameState state;            // 局面
    std::vector<Node*> children;    // 子ノード
    Node* parent;                   // 親ノード
    int n = 0;                      // 訪問回数
    int w = 0;                      // 勝利回数
    float uct = 0.0f;
    bool isExpanded = false;        // 子ノードを展開したかどうか
    dc::moves::Shot shot;           // このノードに至るために打ったショット

    bool IsLeaf() const {
        return children.empty();
    }
};

// UCTの定数
constexpr float UCT_C = 1.414;
int MaxChildrenNum = 2;
bool five_stone_flag = false;
int previous_choice = 0;
std::unordered_map<std::pair<float, float>, std::tuple<float, float, dc::moves::Shot::Rotation>, pair_hash> shot_cache;

std::tuple<float, float, dc::moves::Shot::Rotation> FindOptimalShot(
    float target_x, float target_y);
std::tuple<float, float, dc::moves::Shot::Rotation> GenerateShotCandidates(const dc::GameState& game_state, int choice);
float EvaluateBoard(const dc::GameState& game_state, const dc::GameState& old_state);
std::pair<float, float> IsOpponentStoneInHouse(const dc::GameState& game_state);
bool NumberOneStoneIsMine(const dc::GameState& game_state);
int LeftHasMoreMyStone(const dc::GameState& game_state);

// MCTSの主要関数群
Node* Traverse(Node* node);
Node* BestUCTChild(Node* node);
void CalculateUCTValue(Node* parent, Node* child);
Node* ExpandNode(Node* node);
bool Rollout(Node* node);
void Backpropagate(Node* node, bool result);
float Evaluate(const dc::GameState& state);
float EvaluateGuardStone(Node* node);

// Monte Carlo Tree Searchのエントリポイント
Node* MonteCarloTreeSearch(Node* root, int iterations) {
    //std::cout << "MCTS Start." << "\n";
    for (int i = 0; i < iterations; ++i) {
        Node* leaf = Traverse(root);
        Node* chosen_node = ExpandNode(leaf);
        if (chosen_node->state.IsGameOver()) {
            MaxChildrenNum = iterations;
        }
        bool simulation_result = Rollout(chosen_node);
        Backpropagate(chosen_node, simulation_result);
    }
    return BestUCTChild(root);
}

// 探索フェーズ
Node* Traverse(Node* node) {
    //std::cout << "Traverse Part." << "\n";
    while (node->isExpanded) {
        node = BestUCTChild(node);
    }
    return node;
}

// UCT値が最大の子ノードを選択
Node* BestUCTChild(Node* node) {
    return *std::max_element(node->children.begin(), node->children.end(),
        [](Node* a, Node* b) {
            return a->uct < b->uct;
        });
}

// UCTの計算
void CalculateUCTValue(Node* parent, Node* child) {
    float exploitation = 0.0f;
    if (child == nullptr) return;
    if (child->n == 0) child->uct = std::numeric_limits<float>::infinity();
    if (child->state.shot < 5) {
        exploitation = EvaluateGuardStone(child);
        //std::cout << "Guard Stone Score: " << exploitation << "\n";
    }
    else {
        exploitation = static_cast<float>(child->w) / child->n;
    }
    float exploration = UCT_C * std::sqrt((std::log(parent->n) / child->n));
    child->uct = exploitation + exploration;
}

// 未訪問の子ノードを展開
Node* ExpandNode(Node* node) {
    if (node->isExpanded) return node;
    float speed_x = 0.0f, speed_y = 0.0f, guard_y = 0.0f;
    dc::moves::Shot::Rotation rotation = dc::moves::Shot::Rotation::kCCW;
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist_j(1, 3); // 3, 4, 5, 6手の候補
    std::uniform_int_distribution<int> dist_k(1, 4); // 7, 8手の候補
    int choice = 2;

    // どのショットを選ぶか決定
    if (five_stone_flag) {
        choice = 0;
    }
    else if (node->state.shot < 11) {
        choice = dist_j(gen);
        if (choice == previous_choice) { // 2回連続して同じ候補手にしない処理
            while (choice == previous_choice)
                choice = dist_j(gen);
        }
    }
    else {
        choice = dist_k(gen);
        if (choice == previous_choice) {
            while (choice == previous_choice)
                choice = dist_k(gen);
        }
    }
    //std::cout << "Previous Choice: " << previous_choice << ", New Choice: " << choice << "\n";
    std::tie(speed_x, speed_y, rotation) = GenerateShotCandidates(node->state, choice);
    previous_choice = choice;
    dc::Vector2 shot_velocity(speed_x, speed_y);
    dc::moves::Shot shot{ shot_velocity, rotation };

    dc::GameState new_state = node->state;
    g_simulator->Save(*g_simulator_storage);
    g_simulator->Load(*g_simulator_storage);
    auto& current_player = *g_players[node->state.shot / 4];
    dc::Move move{ shot };
    dc::ApplyMove(g_game_setting, *g_simulator, current_player, new_state, move, std::chrono::milliseconds(0));

    Node* new_node = new Node{ new_state, {}, node, 0, 0, 0.0f, false, shot };
    node->children.push_back(new_node);
    if (node->children.size() >= MaxChildrenNum) {
        node->isExpanded = true;
    }
    return new_node;
}

// ロールアウト（シミュレーション）
bool Rollout(Node* node) {
    Node* current = node;
    int i = node->state.shot;
    dc::GameState new_state = node->state;
    auto& current_player = *g_players[node->state.shot / 4];
    dc::Move temp_move = { node->shot };
    //std::cout << "Rollout Part. shot: " << i << "\n";
    // 後攻最終エンドのラストショット用の処理
    if (node->state.IsGameOver()) {
        bool result = node->state.game_result->winner == g_team;
        //std::cout << "This is the end of the shot. We win?-> " << result << "\n";
        return result;
    }
    // 1手だけシミュレーションを進める
    g_simulator->Load(*g_simulator_storage);
    dc::ApplyMove(g_game_setting, *g_simulator,
        current_player, new_state, temp_move, std::chrono::milliseconds(0));
    g_simulator->Save(*g_simulator_storage);
    float score = EvaluateBoard(new_state, node->state);
    //std::cout << "Board Score: " << score << "\n";
    return  score > 0;
}

// バックプロパゲーション
void Backpropagate(Node* node, bool win) {

    while (node != nullptr) {
        node->n++;
        if (win) node->w++;
        if (node->parent != nullptr) CalculateUCTValue(node->parent, node);
        //std::cout << "Backpropagate: Node " << node << " | n = " << node->n << " | w = " << node->w << " | UCT = " << node->uct << std::endl;
        node = node->parent;
    }
}

// エンド毎の勝敗の評価
float Evaluate(const dc::GameState& state) {
    const dc::GameState& current_state = state;
    dc::Team o_team = dc::GetOpponentTeam(g_team);

    int my_score = current_state.GetTotalScore(g_team);
    int opponent_score = current_state.GetTotalScore(o_team);

    // 自チームのスコア - 相手チームのスコア
    return my_score - opponent_score;
}

// ショットシミュレーション
dc::Vector2 SimulateShot(float vx, float vy, dc::moves::Shot::Rotation rotation) {
    dc::ISimulator::AllStones init_stones;
    init_stones[0].emplace(dc::Vector2(), 0.f, dc::Vector2(vx, vy), rotation == dc::moves::Shot::Rotation::kCCW ? 1.57f : -1.57f);
    auto simulator = dc::simulators::SimulatorFCV1Factory().CreateSimulator();
    simulator->SetStones(init_stones);

    while (!simulator->AreAllStonesStopped()) {
        simulator->Step();
    }

    return simulator->GetStones()[0]->position;
}


void SortStones(std::array<StoneIndex, 16>& result, dc::GameState::Stones const& stones)
{

    for (size_t i = 0; i < 16; ++i)
    {
        result[i].team = i / 8;
        result[i].stone = i % 8;
    }

    for (size_t i = 1; i < 16; ++i)
    {
        for (size_t j = i; j > 0; --j)
        {
            auto const& stone_a = stones[result[j - 1].team][result[j - 1].stone];
            auto const& stone_b = stones[result[j].team][result[j].stone];

            if (!stone_a || (stone_b && (stone_a->position - kCenter).Length() > (stone_b->position - kCenter).Length()))
            {
                std::swap(result[j - 1], result[j]);
            }
        }
    }
}


// ショットの結果が場外でないか確認する関数
bool IsValidPosition(const dc::Vector2& position) {
    constexpr float kMaxX = 2.375f; // リンクの幅
    constexpr float kMaxY = 40.234f; // リンクの長さ

    return std::abs(position.x) < kMaxX &&
        position.y >= 32.004f && position.y <= kMaxY;
}

// ターゲット位置に最適なショットを探索
std::tuple<float, float, dc::moves::Shot::Rotation> FindOptimalShot(
    float target_x, float target_y)
{
    auto target = std::make_pair(target_x, target_y);

    // すでに計算済みならキャッシュを利用
    if (shot_cache.find(target) != shot_cache.end()) {
        //std::cout << "Use Cache" << "\n";
        return shot_cache[target];
    }

    float best_error = std::numeric_limits<float>::max();
    std::tuple<float, float, dc::moves::Shot::Rotation> best_shot;
    // ランダムジェネレータの準備
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<int> rotation_dist(0, 1);
    constexpr int Iteration = 80;
    float base_adjustment = 0.1f;  // 基本の補正量
    float min_adjustment = 0.03f;  // 最小の補正量
    float max_adjustment = 0.15f;  // 最大の補正量

    //float adjustment_factor = 0.09f; // 調整係数
    float target_r = std::sqrt(target_x * target_x + target_y * target_y);
    float target_speed = 0.0f;  // ゴール時の目標速度（微調整可能）
    float v0_speed = 1.122 * 2.1f;

    float vx = v0_speed * (target_x / target_r);
    float vy = v0_speed * (target_y / target_r);

    for (int iter = 0; iter < Iteration; ++iter) {
        // 回転方向をランダムに選択
        dc::moves::Shot::Rotation rotation = rotation_dist(gen) == 0
            ? dc::moves::Shot::Rotation::kCW
            : dc::moves::Shot::Rotation::kCCW;
        auto final_position = SimulateShot(vx, vy, rotation);
        float error_x = target_x - final_position.x;
        float error_y = target_y - final_position.y;

        float error = error_x * error_x + error_y * error_y;

        if (error < best_error) {
            best_error = error;
            best_shot = std::make_tuple(vx, vy, rotation);
            // 誤差が許容範囲内なら終了
            if (error < 0.001f) {
                //std::cout << "Found Optimal Shot and error is " << error << ", iteration=" << iter << "\n";
                break;
            }
        }

        // 動的な補正量を計算（誤差が大きいときは大きく、小さいときは細かく）
        float adjustment_factor = std::clamp(base_adjustment * std::sqrt(error), min_adjustment, max_adjustment);

        // 誤差方向の単位ベクトルを計算
        float error_norm = std::sqrt(error);
        float unit_error_x = error_x / error_norm;
        float unit_error_y = error_y / error_norm;

        // 単位ベクトルに基づいて速度を補正
        vx += adjustment_factor * unit_error_x;
        vy += adjustment_factor * unit_error_y;
    }
    if (target_x == 0.f || target_y == kHouseCenterY || target_y == 2.8f || target_y == 3.0f) {
        shot_cache[target] = best_shot;
    }
    return best_shot;
}


// ガードストーン用の評価関数
float EvaluateGuardStone(Node* node) {
    constexpr float center_weight = 1.0f;   // センターライン重視
    constexpr float cover_weight = 2.0f;    // 相手ストーンを守る効果を重視
    float center_score = 1.0f, cover_score = 0.0f, position_penalty = 0.0f;

    const auto& stones = node->state.stones;
    if (stones.size() == 0) return -100.0f;

    std::vector<dc::Vector2> stone_positions;
    dc::Team o_team = dc::GetOpponentTeam(g_team);
    
    for (int idx = 0; idx < 3; idx++) {
        const auto& guard_stone = stones[static_cast<size_t>(g_team)][idx];
        if (guard_stone) {
            float x = guard_stone->position.x;
            float y = guard_stone->position.y;

            // **1. センターラインに近いほど高評価**
            center_score -= std::abs(x) / kHouseRadius;

            // **2. 相手ストーンを守っているか**
            for (int j = 0; j < 3; j++) {
                const auto& stone = stones[static_cast<size_t>(o_team)][j];
                if (stone && stone->position.y > y) {
                    cover_score += 1.0f; // 相手ストーンの裏なら加点
                }
            }

            // **3. 適切な位置にあるか**
            if (y < kHouseCenterY - 4 * kHouseRadius || y > kHouseCenterY) {
                position_penalty -= 0.5f; // 位置が極端に悪い場合ペナルティ
            }
        }
    }
    return center_weight * center_score + cover_weight * cover_score + position_penalty;
}


// 局面の評価関数
float EvaluateBoard(const dc::GameState& game_state, const dc::GameState& old_state) {
    float score = 0.0f;
    dc::Team o_team = dc::GetOpponentTeam(g_team);
    // ハウスの中心座標
    constexpr dc::Vector2 kCenter(0.0f, kHouseCenterY);

    // ストーンのソート
    std::array<StoneIndex, 16> sorted_stones;
    SortStones(sorted_stones, game_state.stones);

    // ナンバーワンストーンのチェック
    bool my_team_is_closest = false;
    int my_team_stones = 0, opponent_team_stones = 0;
    int num_my_guards = 0, num_opponent_guards = 0;

    for (size_t i = 0; i < 16; ++i) {
        const auto& index = sorted_stones[i];
        const auto& stone = game_state.stones[index.team][index.stone];

        if (!stone) continue; // NULLチェック

        float distance = (stone->position - kCenter).Length();

        // ハウス内かどうか
        if (distance > kHouseRadius) continue;

        // 近さスコア（ハウス中心からの距離を反映）
        float proximity_score = std::max(0.0f, 4.0f - distance);

        // 自分のチームかどうか
        bool is_my_team = (static_cast<size_t>(g_team) == index.team);

        if (is_my_team) {
            score += proximity_score;
            if (i == 0) my_team_is_closest = true;
            my_team_stones++;

            // ガードストーンのカウント（ハウス手前にあるもの）
            if (stone->position.y < kHouseCenterY - 1.5f) {
                num_my_guards++;
            }
        }
        else {
            score -= proximity_score;
            opponent_team_stones++;

            // 相手のガードストーンカウント
            if (stone->position.y < kHouseCenterY - 1.5f) {
                num_opponent_guards++;
            }
        }
    }

    //  **相手のストーンを弾き出した場合のボーナス**
    if (old_state.shot >= 12) {
        int g_new = 0, g_old = 0, o_new = 0, o_old = 0;
        for (size_t i = 0; i < 8; i++) {
            const auto& g_new_stone = game_state.stones[static_cast<size_t>(g_team)][i];
            const auto& g_old_stone = old_state.stones[static_cast<size_t>(g_team)][i];
            const auto& o_new_stone = game_state.stones[static_cast<size_t>(o_team)][i];
            const auto& o_old_stone = old_state.stones[static_cast<size_t>(o_team)][i];

            if (std::pow(g_new_stone->position.x, 2) + std::pow(g_new_stone->position.y, 2) <= std::pow(kHouseRadius, 2)) g_new++;
            if (std::pow(g_old_stone->position.x, 2) + std::pow(g_old_stone->position.y, 2) <= std::pow(kHouseRadius, 2)) g_old++;
            if (std::pow(o_new_stone->position.x, 2) + std::pow(o_new_stone->position.y, 2) <= std::pow(kHouseRadius, 2)) o_new++;
            if (std::pow(o_old_stone->position.x, 2) + std::pow(o_old_stone->position.y, 2) <= std::pow(kHouseRadius, 2)) o_old++;
        }

        if (g_new > g_old) {
            score += 10.0f; // 自分のストーンが増えたらボーナス
        }
        else if (g_new < g_old) {
            score -= 5.0f; //  自分のストーンが消えたら減点
        }
        if (o_new > o_old) {
            score -= 5.0f; // 相手のストーンが増えたら減点
        }
        else if (o_new < o_old) {
            score += 10.0f; //  相手のストーンが消えたらボーナス
        }
    }

    // 【ボーナス: ナンバーワンストーンが自チームなら加点】
    if (my_team_is_closest) {
        score += 20.0f;
    }

    // 【ストーン数によるボーナス】
    if (my_team_stones > opponent_team_stones) {
        score += 3.0f;
    }

    // 【ガードストーンの評価】
    score += num_my_guards * 2.0f;   // 自分のガードストーンが多いほど良い
    score -= num_opponent_guards * 2.0f; // 相手のガードが多いと不利

    // 【相手のストーンを弾いたボーナス】（自分の石が増え & 相手の石が減った）
    score += (my_team_stones - opponent_team_stones) * 5.0f;

    return score;
}


// 相手ストーンの重心を計算.この関数はガードストーンを置いた後に呼ぶ.
std::pair<float, float> GetOpponentGravityPos(const dc::GameState& game_state) {
    dc::Team o_team = dc::GetOpponentTeam(g_team);
    const auto& stones = game_state.stones;

    std::vector<dc::Vector2> stone_positions;
    int num_stones = 0;
    float sum_x = 0.0f, sum_y = 0.0f;
    for (int idx = 0; idx < 8; idx++) {
        const auto& stone = stones[static_cast<size_t>(o_team)][idx];
        if (stone) {
            // 重心を求める
            sum_x += stone->position.x;
            sum_y += stone->position.y;
            num_stones++;
        }
    }
    float centroid_x = sum_x / num_stones;
    float centroid_y = sum_y / num_stones;

    return { centroid_x, centroid_y };
}

std::pair<float, float> IsOpponentStoneInHouse(const dc::GameState& game_state) {
    dc::Team o_team = dc::GetOpponentTeam(g_team);
    const auto& stones = game_state.stones;

    std::pair<float, float> closest_stone = { kHouseRadius, kHouseCenterY };
    float min_distance = std::numeric_limits<float>::max();

    for (int idx = 0; idx < 8; idx++) {
        const auto& stone = stones[static_cast<size_t>(o_team)][idx];
        if (stone) {
            float x = stone->position.x;
            float y = stone->position.y;

            // ストーンとハウス中心の距離を計算
            float distance = std::pow(x, 2) + std::pow(y - kHouseCenterY, 2);

            // ハウス内にある場合のみ比較
            if (distance <= std::pow(kHouseRadius / 3.0f, 2) && distance < min_distance) {
                min_distance = distance;
                closest_stone = { x, y };
            }
        }
    }

    return closest_stone;
}

bool NumberOneStoneIsMine(const dc::GameState& game_state) {
    const auto& stones = game_state.stones;
    float min_distance = std::numeric_limits<float>::max();
    size_t owner = 0;
    for (size_t team = 0; team < 2; team++) {
        for (int idx = 0; idx < 8; idx++) {
            const auto& stone = stones[(team)][idx];
            if (stone) {
                float x = stone->position.x;
                float y = stone->position.y;

                // ハウス内のストーンのみ対象
                if (std::hypot(x, y - kHouseCenterY) <= kHouseRadius) {
                    float distance = std::hypot(x, y - kHouseCenterY);
                    if (distance < min_distance) {
                        min_distance = distance;
                        owner = team;
                    }
                }
            }
        }
    }

    return owner == static_cast<size_t>(g_team); // 自チームのストーンがナンバーワンなら true

}

int LeftHasMoreMyStone(const dc::GameState& game_state) {
    const auto& stones = game_state.stones;
    //dc::Team o_team = dc::GetOpponentTeam(g_team);
    int left = 0, right = 0;
    
    for (int i = 0; i < 8; i++) {
        const auto& stone = stones[static_cast<size_t>(g_team)][i];
        if (stone && stone->position.x >= 0) right++;
        else if (stone && stone->position.x < 0) left++;
        else break;
    }
    return left - right;
}


// 局面を展開するための候補手の生成
std::tuple<float, float, dc::moves::Shot::Rotation> GenerateShotCandidates(const dc::GameState& game_state, int choice) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<int> rotation_dist(0, 1); // 0 or 1 for rotation.
    std::uniform_real_distribution<> dist_f(kHouseRadius, 2.7f); // r~2.7mの範囲でランダム選択
    std::uniform_real_distribution<> dist_x(kHouseRadius / 3.0f, kHouseRadius * 2.0f / 3.0f); // r/3〜2r/3の範囲でランダム選択.コーナーガード用

    float distance = dist_f(gen);

    float vx = 0.f, vy = 0.f;
    // 回転方向をランダムに選択
    dc::moves::Shot::Rotation rotation = rotation_dist(gen) == 0
        ? dc::moves::Shot::Rotation::kCW
        : dc::moves::Shot::Rotation::kCCW;

    switch (choice) {
        case 0: { // ガードストーンを置く
            //std::cout << "Choose 0: Guard Stone" << "\n";
            auto guard_y = kHouseCenterY - distance; // ハウスから4m手前
            if (g_team != game_state.hammer) { // 先行の場合、ガードストーンは3つ投げられる.
                if (game_state.shot == 0) { // ガード
                    std::uniform_real_distribution<> dist_f(kHouseRadius * 2.0f / 3.0f, kHouseRadius * 1.5);
                    guard_y = kHouseCenterY - dist_f(gen);
                    auto best_0_shot = FindOptimalShot(0.f, guard_y);
                    std::tie(vx, vy, rotation) = best_0_shot;
                }
                else if (game_state.shot == 2) { // ガード
                    std::uniform_real_distribution<> dist_f(kHouseRadius * 1.5, kHouseRadius * 2);
                    guard_y = kHouseCenterY - dist_f(gen);
                    auto best_0_shot = FindOptimalShot(0.f, guard_y);
                    std::tie(vx, vy, rotation) = best_0_shot;
                }
                else if (game_state.shot == 4) { // 攻撃
                    auto stone_pos = IsOpponentStoneInHouse(game_state);
                    if (stone_pos.first != kHouseRadius || stone_pos.second != kHouseCenterY) {
                        float x = stone_pos.first;
                        float y = stone_pos.second;
                        std::tie(vx, vy, rotation) = FindOptimalShot(-x, y);
                    }
                    else {
                        auto best_0_shot = FindOptimalShot(0.f, kHouseCenterY);
                        std::tie(vx, vy, rotation) = best_0_shot;
                    }
                }
            }
            else { // 後攻の場合、ガードストーンは2つ投げられる.
                // できればテイクアウト, だめならコーナーガード
                auto stone_pos = IsOpponentStoneInHouse(game_state);
                if (stone_pos.first != kHouseRadius || stone_pos.second != kHouseCenterY) {
                    float x = stone_pos.first;
                    float y = stone_pos.second;
                    std::tie(vx, vy, rotation) = FindOptimalShot(-x, y);
                }
                else {
                    auto guard_x = dist_x(gen);
                    if (guard_x  * LeftHasMoreMyStone(game_state) > 0) guard_x = -guard_x;
                    auto best_0_shot = FindOptimalShot(guard_x, guard_y);
                    std::tie(vx, vy, rotation) = best_0_shot;
                }
            }
            break;
        }
        case 1: { // 重心による判断
            //std::cout << "Choose 1: Gravity Position" << "\n";
            std::pair<float, float> gravity_pos = GetOpponentGravityPos(game_state);
            float x = gravity_pos.first;
            float y = gravity_pos.second;
            if (y <= kHouseCenterY) { // hide
                auto best_1_shot = FindOptimalShot(x, kHouseCenterY + std::min(std::abs(kHouseCenterY - y), kHouseRadius / 3.0f));
                std::tie(vx, vy, rotation) = best_1_shot;
            }
            else { // attack
                auto best_1_shot = FindOptimalShot(x, std::max(y, kHouseCenterY - kHouseRadius / 3.0f));
                std::tie(vx, vy, rotation) = best_1_shot;
            }
            break;
        }
        case 2: { // ボタン狙い
            //std::cout << "Choose 2: Button Aim" << "\n";
            auto best_2_shot = FindOptimalShot(0.f, kHouseCenterY);
            std::tie(vx, vy, rotation) = best_2_shot;
            break;
        }
        case 3: { // テイクアウト(ハウス内)
            //std::cout << "Take Out!" << "\n";
            auto stone_pos = IsOpponentStoneInHouse(game_state);
            if (stone_pos.first != kHouseRadius || stone_pos.second != kHouseCenterY) {
                float x = stone_pos.first;
                float y = stone_pos.second;
                std::tie(vx, vy, rotation) = FindOptimalShot(-x, y);
            }
            else {
                //std::cout << "Failed to find the opponent stone in house." << "\n";
                std::tie(vx, vy, rotation) = FindOptimalShot(0.f, kHouseCenterY);
            }
            break;
        }
        case 4: { // 高速球(ほぼ直線). 衝突なしなら飛び出る.
            dc::Team o_team = dc::GetOpponentTeam(g_team);
            if (game_state.GetTotalScore(g_team) <= game_state.GetTotalScore(o_team)) {
                //std::cout << "Break The Board!!" << "\n";
                if (LeftHasMoreMyStone(game_state) > 0) {
                    vx = -0.1f; vy = 2.8f; rotation = dc::moves::Shot::Rotation::kCW;
                }
                else if(LeftHasMoreMyStone(game_state) < 0){
                    vx = 0.1f; vy = 2.8f; rotation = dc::moves::Shot::Rotation::kCCW;
                }
                else {
                    std::tie(vx, vy, rotation) = FindOptimalShot(0.f, kHouseCenterY);
                }
            }
            else {
                //std::cout << "Destroy Surrounding Stones" << "\n";
                if (LeftHasMoreMyStone(game_state) > 0) {
                    vx = 0.0f; vy = 3.0f; rotation = dc::moves::Shot::Rotation::kCCW;
                }
                else if(LeftHasMoreMyStone(game_state) < 0){
                    vx = 0.0f; vy = 3.0f; rotation = dc::moves::Shot::Rotation::kCW;
                }
                else {
                    std::tie(vx, vy, rotation) = FindOptimalShot(0.f, kHouseCenterY);
                }
            }
            break;
        }
    }

    return std::make_tuple(vx, vy, rotation);
}

/// \brief サーバーから送られてきた試合設定が引数として渡されるので，試合前の準備を行います．
///
/// 引数 \p player_order を編集することでプレイヤーのショット順を変更することができます．各プレイヤーの情報は \p player_factories に格納されています．
/// 補足：プレイヤーはショットのブレをつかさどります．プレイヤー数は4で，0番目は0, 1投目，1番目は2, 3投目，2番目は4, 5投目，3番目は6, 7投目を担当します．
///
/// この処理中の思考時間消費はありません．試合前に時間のかかる処理を行う場合この中で行うべきです．
///
/// \param team この思考エンジンのチームID．
///     Team::k0 の場合，最初のエンドの先攻です．
///     Team::k1 の場合，最初のエンドの後攻です．
///
/// \param game_setting 試合設定．
///     この参照はOnInitの呼出し後は無効になります．OnInitの呼出し後にも参照したい場合はコピーを作成してください．
///
/// \param simulator_factory 試合で使用されるシミュレータの情報．
///     未対応のシミュレータの場合 nullptr が格納されます．
///
/// \param player_factories 自チームのプレイヤー情報．
///     未対応のプレイヤーの場合 nullptr が格納されます．
///
/// \param player_order 出力用引数．
///     プレイヤーの順番(デフォルトで0, 1, 2, 3)を変更したい場合は変更してください．
void OnInit(
    dc::Team team,
    dc::GameSetting const& game_setting,
    std::unique_ptr<dc::ISimulatorFactory> simulator_factory,
    std::array<std::unique_ptr<dc::IPlayerFactory>, 4> player_factories,
    std::array<size_t, 4> & player_order)
{
    // TODO AIを作る際はここを編集してください

    g_team = team;
    g_game_setting = game_setting;
    if (simulator_factory) {
        g_simulator = simulator_factory->CreateSimulator(); // simulator 生成 
    }
    else {
        g_simulator = dc::simulators::SimulatorFCV1Factory().CreateSimulator();
    }
    g_simulator_storage = g_simulator->CreateStorage();

    // プレイヤーを生成する
    // 非対応の場合は NormalDistプレイヤーを使用する．
    assert(g_players.size() == player_factories.size());
    for (size_t i = 0; i < g_players.size(); ++i) {
        auto const& player_factory = player_factories[player_order[i]];
        if (player_factory) {
            g_players[i] = player_factory->CreatePlayer();
        }
        else {
            g_players[i] = dc::players::PlayerNormalDistFactory().CreatePlayer();
        }

    }
}

/// \brief 自チームのターンに呼ばれます．返り値として返した行動がサーバーに送信されます．
///
/// \param game_state 現在の試合状況．
///     この参照は関数の呼出し後に無効になりますので，関数呼出し後に参照したい場合はコピーを作成してください．
///
/// \return 選択する行動．この行動が自チームの行動としてサーバーに送信されます．
dc::Move OnMyTurn(dc::GameState const& game_state)
{
    // TODO AIを作る際はここを編集してください

    dc::moves::Shot shot;
    float vx = 0.0f, vy = 0.0f;
    dc::moves::Shot::Rotation rotation = dc::moves::Shot::Rotation::kCCW;
    if (game_state.shot < 5) {
        five_stone_flag = true;
    }
    else five_stone_flag = false;

    dc::GameState initial_state = game_state;
    // 親ノード
    Node* root = new Node{ initial_state, {}, nullptr, 0, 0, 0.0f, false, {} };
    int iterations = 4;  // 探索回数
    if (game_state.shot < 5) {
        iterations = 1;
    }
    else if (game_state.shot < 11) {
        iterations = 4;
    }
    else {
        iterations = 4;
    }
    // MCTS実行
    Node* best_node = MonteCarloTreeSearch(root, iterations);
    //std::cout << "Root has " << root->children.size() << " children." << "\n";
    dc::moves::Shot best_shot = best_node->shot;
    // 最良のショットを表示
    //std::cout << "Best Node: " << best_node << "\n";
    //std::cout << "Best shot: SpeedX = " << best_shot.velocity.x
    //    << ", SpeedY = " << best_shot.velocity.y
    //    << ", Rotation = " << (best_shot.rotation == dc::moves::Shot::Rotation::kCW ? "CW" : "CCW")
    //    << std::endl;
    // メモリ解放
    delete root;

    shot.velocity.x = best_shot.velocity.x;
    shot.velocity.y = best_shot.velocity.y;
    shot.rotation = best_shot.rotation;

    return shot;
}

/// \brief 相手チームのターンに呼ばれます．AIを作る際にこの関数の中身を記述する必要は無いかもしれません．
///
/// ひとつ前の手番で自分が行った行動の結果を見ることができます．
///
/// \param game_state 現在の試合状況．
///     この参照は関数の呼出し後に無効になりますので，関数呼出し後に参照したい場合はコピーを作成してください．
void OnOpponentTurn(dc::GameState const& game_state)
{
    // TODO AIを作る際はここを編集してください
}



/// \brief ゲームが正常に終了した際にはこの関数が呼ばれます．
///
/// \param game_state 試合終了後の試合状況．
///     この参照は関数の呼出し後に無効になりますので，関数呼出し後に参照したい場合はコピーを作成してください．
void OnGameOver(dc::GameState const& game_state)
{
    // TODO AIを作る際はここを編集してください

    if (game_state.game_result->winner == g_team) {
        std::cout << "won the game" << std::endl;
    } else {
        std::cout << "lost the game" << std::endl;
    }
}



} // unnamed namespace



int main(int argc, char const * argv[])
{
    using boost::asio::ip::tcp;
    using nlohmann::json;

    // TODO AIの名前を変更する場合はここを変更してください．
    constexpr auto kName = "MCTSweeper";

    constexpr int kSupportedProtocolVersionMajor = 1;

    try {
        if (argc != 3) {
            std::cerr << "Usage: command <host> <port>" << std::endl;
            return 1;
        }

        boost::asio::io_context io_context;

        tcp::socket socket(io_context);
        tcp::resolver resolver(io_context);
        boost::asio::connect(socket, resolver.resolve(argv[1], argv[2]));  // 引数のホスト，ポートに接続します．

        // ソケットから1行読む関数です．バッファが空の場合，新しい行が来るまでスレッドをブロックします．
        auto read_next_line = [&socket, input_buffer = std::string()] () mutable {
            // read_untilの結果，input_bufferに複数行入ることがあるため，1行ずつ取り出す処理を行っている
            if (input_buffer.empty()) {
                boost::asio::read_until(socket, boost::asio::dynamic_buffer(input_buffer), '\n');
            }
            auto new_line_pos = input_buffer.find_first_of('\n');
            auto line = input_buffer.substr(0, new_line_pos + 1);
            input_buffer.erase(0, new_line_pos + 1);
            return line;
        };

        // コマンドが予期したものかチェックする関数です．
        auto check_command = [] (nlohmann::json const& jin, std::string_view expected_cmd) {
            auto const actual_cmd = jin.at("cmd").get<std::string>();
            if (actual_cmd != expected_cmd) {
                std::ostringstream buf;
                buf << "Unexpected cmd (expected: \"" << expected_cmd << "\", actual: \"" << actual_cmd << "\")";
                throw std::runtime_error(buf.str());
            }
        };

        dc::Team team = dc::Team::kInvalid;

        // [in] dc
        {
            auto const line = read_next_line();
            auto const jin = json::parse(line);

            check_command(jin, "dc");

            auto const& jin_version = jin.at("version");
            if (jin_version.at("major").get<int>() != kSupportedProtocolVersionMajor) {
                throw std::runtime_error("Unexpected protocol version");
            }

            std::cout << "[in] dc" << std::endl;
            std::cout << "  game_id  : " << jin.at("game_id").get<std::string>() << std::endl;
            std::cout << "  date_time: " << jin.at("date_time").get<std::string>() << std::endl;
        }

        // [out] dc_ok
        {
            json const jout = {
                { "cmd", "dc_ok" },
                { "name", kName }
            };
            auto const output_message = jout.dump() + '\n';
            boost::asio::write(socket, boost::asio::buffer(output_message));

            std::cout << "[out] dc_ok" << std::endl;
            std::cout << "  name: " << kName << std::endl;
        }


        // [in] is_ready
        {
            auto const line = read_next_line();
            auto const jin = json::parse(line);

            check_command(jin, "is_ready");

            if (jin.at("game").at("rule").get<std::string>() != "normal") {
                throw std::runtime_error("Unexpected rule");
            }

            team = jin.at("team").get<dc::Team>();

            auto const game_setting = jin.at("game").at("setting").get<dc::GameSetting>();

            auto const& jin_simulator = jin.at("game").at("simulator");
            std::unique_ptr<dc::ISimulatorFactory> simulator_factory;
            try {
                simulator_factory = jin_simulator.get<std::unique_ptr<dc::ISimulatorFactory>>();
            } catch (std::exception & e) {
                std::cout << "Exception: " << e.what() << std::endl;
            }

            auto const& jin_player_factories = jin.at("game").at("players").at(dc::ToString(team));
            std::array<std::unique_ptr<dc::IPlayerFactory>, 4> player_factories;
            for (size_t i = 0; i < 4; ++i) {
                std::unique_ptr<dc::IPlayerFactory> player_factory;
                try {
                    player_factory = jin_player_factories[i].get<std::unique_ptr<dc::IPlayerFactory>>();
                } catch (std::exception & e) {
                    std::cout << "Exception: " << e.what() << std::endl;
                }
                player_factories[i] = std::move(player_factory);
            }

            std::cout << "[in] is_ready" << std::endl;
        
        // [out] ready_ok

            std::array<size_t, 4> player_order{ 0, 1, 2, 3 };
            OnInit(team, game_setting, std::move(simulator_factory), std::move(player_factories), player_order);

            json const jout = {
                { "cmd", "ready_ok" },
                { "player_order", player_order }
            };
            auto const output_message = jout.dump() + '\n';
            boost::asio::write(socket, boost::asio::buffer(output_message));

            std::cout << "[out] ready_ok" << std::endl;
            std::cout << "  player order: " << jout.at("player_order").dump() << std::endl;
        }

        // [in] new_game
        {
            auto const line = read_next_line();
            auto const jin = json::parse(line);

            check_command(jin, "new_game");

            std::cout << "[in] new_game" << std::endl;
            std::cout << "  team 0: " << jin.at("name").at("team0") << std::endl;
            std::cout << "  team 1: " << jin.at("name").at("team1") << std::endl;
        }

        dc::GameState game_state;

        while (true) {
            // [in] update
            auto const line = read_next_line();
            auto const jin = json::parse(line);

            check_command(jin, "update");

            game_state = jin.at("state").get<dc::GameState>();

            std::cout << "[in] update (end: " << int(game_state.end) << ", shot: " << int(game_state.shot) << ")" << std::endl;

            // if game was over
            if (game_state.game_result) {
                break;
            }

            if (game_state.GetNextTeam() == team) { // my turn
                // [out] move
                auto move = OnMyTurn(game_state);
                json jout = {
                    { "cmd", "move" },
                    { "move", move }
                };
                auto const output_message = jout.dump() + '\n';
                boost::asio::write(socket, boost::asio::buffer(output_message));
                
                std::cout << "[out] move" << std::endl;
                if (std::holds_alternative<dc::moves::Shot>(move)) {
                    dc::moves::Shot const& shot = std::get<dc::moves::Shot>(move);
                    std::cout << "  type    : shot" << std::endl;
                    std::cout << "  velocity: [" << shot.velocity.x << ", " << shot.velocity.y << "]" << std::endl;
                    std::cout << "  rotation: " << (shot.rotation == dc::moves::Shot::Rotation::kCCW ? "ccw" : "cw") << std::endl;
                } else if (std::holds_alternative<dc::moves::Concede>(move)) {
                    std::cout << "  type: concede" << std::endl;
                }

            } else { // opponent turn
                OnOpponentTurn(game_state);
            }
        }

        // [in] game_over
        {
            auto const line = read_next_line();
            auto const jin = json::parse(line);

            check_command(jin, "game_over");

            std::cout << "[in] game_over" << std::endl;
        }

        // 終了．
        OnGameOver(game_state);

    } catch (std::exception & e) {
        std::cerr << "Exception: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "Unknown exception" << std::endl;
    }

    return 0;
}