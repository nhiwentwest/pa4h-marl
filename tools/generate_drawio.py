import os
import shutil
import subprocess

class DrawioDiagram:
    def __init__(self):
        self.cells = []
        # Base cells required for any draw.io model
        self.cells.append('<mxCell id="0" />')
        self.cells.append('<mxCell id="1" parent="0" />')
        self.cell_count = 2

    def next_id(self):
        self.cell_count += 1
        return str(self.cell_count)

    def add_vertex(self, cell_id, value, style, x, y, w, h):
        # Properly escape XML entities to avoid parsing failures, while allowing Draw.io to render HTML
        val_esc = value.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace('\n', '&lt;br/&gt;')
        cell_str = f'<mxCell id="{cell_id}" value="{val_esc}" style="{style}" vertex="1" parent="1">'
        cell_str += f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry" />'
        cell_str += '</mxCell>'
        self.cells.append(cell_str)
        return cell_id

    def add_edge(self, cell_id, source, target, style, value="", entry_exit="", points=None):
        val_esc = value.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace('\n', '&lt;br/&gt;')
        full_style = style
        if entry_exit:
            full_style += f";{entry_exit}"
        cell_str = f'<mxCell id="{cell_id}" value="{val_esc}" style="{full_style}" edge="1" parent="1" source="{source}" target="{target}">'
        cell_str += '<mxGeometry relative="1" as="geometry">'
        if points:
            cell_str += '<Array as="points">'
            for pt in points:
                cell_str += f'<mxPoint x="{pt[0]}" y="{pt[1]}" />'
            cell_str += '</Array>'
        cell_str += '</mxGeometry>'
        cell_str += '</mxCell>'
        self.cells.append(cell_str)
        return cell_id

    def to_xml(self, page_name="Page 1"):
        xml = '<mxfile host="Electron" modified="2026-07-13T12:00:00.000Z" agent="Codex" version="21.6.8" type="device">\n'
        xml += f'  <diagram id="{page_name.lower().replace(" ", "_")}" name="{page_name}">\n'
        xml += '    <mxGraphModel dx="1200" dy="1000" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="827" pageHeight="1169" math="0" shadow="0">\n'
        xml += '      <root>\n'
        for cell in self.cells:
            xml += f'        {cell}\n'
        xml += '      </root>\n'
        xml += '    </mxGraphModel>\n'
        xml += '  </diagram>\n'
        xml += '</mxfile>\n'
        return xml

def build_fig1_legacy():
    d = DrawioDiagram()
    
    # Muted, professional academic palette (grayscale / neutral-based)
    font_base = "fontFamily=Helvetica;fontSize=11;fontColor=#333333;"
    
    # White fill, thin grey border for general components
    style_neutral = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#555555;strokeWidth=1.2;{font_base}align=center;"
    
    # Light grey fill, dark grey border for key active agents
    style_agent = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#f5f5f5;strokeColor=#333333;strokeWidth=1.5;{font_base}align=center;fontWeight=bold;"
    
    # Dotted/dashed zones - Added labelBackgroundColor=#ffffff to overlap lines cleanly and shifted slightly right
    style_zone = f"rounded=0;dashed=1;dashPattern=8 8;fillColor=none;strokeColor=#888888;strokeWidth=1.2;html=1;{font_base}align=right;verticalAlign=top;spacingRight=5;spacingTop=10;labelBackgroundColor=#ffffff;"
    
    # Muted arrows (No colors used in Fig 1, completely grayscale/black)
    style_edge_solid = "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;endArrow=classic;strokeWidth=1.2;strokeColor=#333333;"
    style_edge_dashed = "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;dashed=1;endArrow=classic;strokeWidth=1.2;strokeColor=#666666;"

    # 1. Data Sources
    d.add_vertex("alibaba", "Alibaba GPU Trace\n• Arrival & duration\n• Node & priority requirements", style_neutral, 100, 100, 200, 80)
    d.add_vertex("nrel", "NREL GPU Power\n• 0.2-second traces\n• Multi-rate power dynamics", style_neutral, 400, 100, 200, 80)
    
    # 2. Matching
    d.add_vertex("matching", "Workload–Power Profile Matching\n• Temporal matching of GPU demands", style_neutral, 250, 220, 220, 70)
    
    # 3. Python Gang Environment (Zone)
    d.add_vertex("zone_env", "Python Gang Environment (RL Interface)", style_zone, 100, 330, 520, 240)
    
    # Inside zone_env: Queue, A4, A2, A3, A1
    d.add_vertex("queue", "Queue\n(Jobs)", style_neutral, 120, 380, 80, 50)
    d.add_vertex("a4", "A4 Placement\nST-GNN", style_agent, 240, 380, 100, 50)
    d.add_vertex("a2", "A2 Risk Detector\nST-GNN", style_agent, 410, 380, 100, 50)
    d.add_vertex("a3", "A3 Victim Selector\nMLP", style_agent, 410, 480, 100, 50)
    d.add_vertex("a1", "A1 Wait/Preempt\nMLP", style_agent, 240, 480, 100, 50)
    
    # 4. Gang Cluster Simulator (Zone)
    d.add_vertex("zone_sim", "Gang Cluster Simulator (Custom Gym/Py4J Engine)", style_zone, 100, 610, 520, 360)
    
    # Inside zone_sim: Layers
    d.add_vertex("host_layer", "Host Layer\n• 64 GPU hosts | • Exclusive gang allocation", style_neutral, 130, 660, 460, 50)
    d.add_vertex("rack_layer", "Rack / Power Layer\n• 16 racks x 4 hosts | • Power budget & oversubscription\n• 1,500 power samples per RL step (0.2s rate)", style_neutral, 130, 730, 460, 65)
    d.add_vertex("network_layer", "Network Layer\n• Ring All-Reduce packet flows (shared fat-tree link model)\n• Host-Edge-Aggregate-Edge-Host | • Congestion, latency, BW", style_neutral, 130, 815, 460, 65)
    d.add_vertex("sla_layer", "SLA Layer\n• Admission/Completion deadlines\n• Checkpoint-restart overhead model", style_neutral, 130, 900, 460, 50)

    # 5. Metrics & Rewards
    d.add_vertex("metrics", "Metrics and Multi-objective Reward\n• (+) served / goodput\n• (-) energy & rack power violations\n• (-) waiting & checkpoint cost\n• (-) network delay & SLA violations", style_neutral, 100, 1010, 520, 130)

    # 6. CTDE-PPO Training
    d.add_vertex("training", "CTDE–PPO Training\n• A2/A4 rack ST-GNNs; A1/A3 MLPs; centralized MLP critic\n• GAE + PPO + counterfactual auxiliary loss\n• SLA dual-ascent multipliers", style_neutral, 100, 1180, 520, 110)

    # SLA Multipliers box (Thin grey outline for neutrality)
    d.add_vertex("multipliers", "SLA Dual-Ascent\nMultipliers\n(\u03bb_adm, \u03bb_comp)", style_neutral, 660, 1050, 130, 60)

    # Edges
    d.add_edge("e_ali", "alibaba", "matching", style_edge_solid)
    d.add_edge("e_nrel", "nrel", "matching", style_edge_solid)
    d.add_edge("e_match", "matching", "queue", style_edge_solid)
    
    # Environment flow connections
    d.add_edge("e_q_a4", "queue", "a4", style_edge_solid)
    d.add_edge("e_a4_a2", "a4", "a2", style_edge_solid)
    d.add_edge("e_a2_a3", "a2", "a3", style_edge_solid, "flagged rack")
    d.add_edge("e_a3_a1", "a3", "a1", style_edge_solid, "candidate")
    d.add_edge("e_a1_out", "a1", "host_layer", style_edge_solid, "placement / preemption")
    
    # Simulator internally connects (represented as layered flow down)
    d.add_edge("e_l1", "host_layer", "rack_layer", style_edge_solid)
    d.add_edge("e_l2", "rack_layer", "network_layer", style_edge_solid)
    d.add_edge("e_l3", "network_layer", "sla_layer", style_edge_solid)
    
    # Simulator to metrics
    d.add_edge("e_sim_met", "sla_layer", "metrics", style_edge_solid)
    
    # Metrics to training
    d.add_edge("e_met_trn", "metrics", "training", style_edge_solid, "team reward + state")
    
    # Feedback loops - ROUTED OUTSIDE to avoid overlapping boxes!
    # 1. State feedback loop to Env (Outer Left loop)
    d.add_edge("fb_state", "metrics", "zone_env", style_edge_solid, "next state", 
               "exitX=0;exitY=0.75;entryX=0;entryY=0.25;", 
               points=[(40, 1108), (40, 390)])
    
    # 2. Dashed gradients to Centralized Critic (Outer Right loop)
    d.add_edge("fb_critic", "training", "metrics", style_edge_dashed, "critic training only", 
               "exitX=1;exitY=0.25;entryX=1;entryY=0.75;",
               points=[(645, 1208), (645, 1108)])
               
    # 3. Dashed PPO update to Actors (Outer Left loop, nested inside fb_state)
    d.add_edge("fb_actors", "training", "zone_env", style_edge_dashed, "PPO update / gradients", 
               "exitX=0;exitY=0.25;entryX=0;entryY=0.75;",
               points=[(20, 1208), (20, 510)])
               
    # 4. SLA feedback to dual ascent
    d.add_edge("sla_to_mult", "metrics", "multipliers", style_edge_solid, "violations", "exitX=1;exitY=0.5;entryX=0;entryY=0.5;")
    
    # 5. SLA multipliers feedback to training (Routed cleanly around)
    d.add_edge("mult_to_trn", "multipliers", "training", style_edge_dashed, "\u03bb feedback",
               "exitX=0.5;exitY=1;entryX=1;entryY=0.5;",
               points=[(725, 1235)])

    return d.to_xml("End-to-End System Architecture")


def build_fig1():
    """Landscape system overview sized for a two-column paper."""
    d = DrawioDiagram()
    font = "fontFamily=Helvetica;fontSize=11;fontColor=#333333;"
    neutral = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#555555;strokeWidth=1.2;{font}align=center;"
    agent = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#f5f5f5;strokeColor=#333333;strokeWidth=1.5;{font}align=center;fontWeight=bold;"
    zone = f"rounded=0;dashed=1;dashPattern=8 8;fillColor=none;strokeColor=#888888;strokeWidth=1.2;html=1;{font}align=right;verticalAlign=top;spacingRight=8;spacingTop=8;labelBackgroundColor=#ffffff;"
    solid = "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;endArrow=classic;strokeWidth=1.2;strokeColor=#333333;"
    dashed = solid + "dashed=1;strokeColor=#666666;"

    d.add_vertex("workloads", "Workloads", neutral, 55, 70, 220, 70)
    d.add_vertex("matching", "Common job schema\n+ workload–power matching", neutral, 50, 200, 230, 62)

    d.add_vertex("zone_env", "Python Gang Environment / RL Interface", zone, 315, 40, 520, 300)
    d.add_vertex("queue", "Pending\nqueue", neutral, 345, 105, 85, 52)
    d.add_vertex("a4", "A4 Placement\nST-GNN", agent, 470, 105, 120, 52)
    d.add_vertex("a2", "A2 Risk\nST-GNN", agent, 655, 105, 120, 52)
    d.add_vertex("a3", "A3 Victim\nMLP", agent, 655, 235, 120, 52)
    d.add_vertex("a1", "A1 Wait/Preempt\nMLP", agent, 470, 235, 120, 52)

    d.add_vertex("zone_sim", "Gang Cluster Simulator / Py4J", zone, 880, 40, 430, 430)
    d.add_vertex("host", "Host layer\n64 exclusive GPU hosts", neutral, 920, 100, 350, 52)
    d.add_vertex("rack", "Rack / power layer\n16 racks × 4 hosts; 1,500 samples/step", neutral, 920, 180, 350, 58)
    d.add_vertex("network", "Network layer\nring all-reduce; shared fat-tree links", neutral, 920, 270, 350, 58)
    d.add_vertex("sla", "SLA layer\nadmission/completion; checkpoint–restart", neutral, 920, 360, 350, 58)

    d.add_vertex("metrics", "Metrics and multi-objective reward\nserved workload; energy; rack violations; waiting; checkpoint; network; SLA", neutral, 315, 395, 520, 72)
    d.add_vertex("training", "CTDE–PPO training\nA2/A4 ST-GNNs; A1/A3 MLPs; centralized MLP critic\nGAE + PPO + counterfactual loss", neutral, 315, 535, 520, 86)
    d.add_vertex("dual", "SLA dual ascent\nλ_adm, λ_comp", neutral, 940, 535, 190, 62)

    d.add_edge("e1", "workloads", "matching", solid)
    d.add_edge("e3", "matching", "queue", solid)
    d.add_edge("e4", "queue", "a4", solid)
    d.add_edge("e5", "a4", "a2", solid)
    d.add_edge("e6", "a2", "a3", solid, "flagged rack")
    d.add_edge("e7", "a3", "a1", solid, "", "exitX=0;exitY=0.5;entryX=1;entryY=0.5;")
    d.add_edge("e8", "a1", "host", solid, "", "exitX=0.5;exitY=1;entryX=0;entryY=0.5;", points=[(530, 360), (850, 360), (850, 126)])
    d.add_edge("e9", "host", "rack", solid)
    d.add_edge("e10", "rack", "network", solid)
    d.add_edge("e11", "network", "sla", solid)
    d.add_edge("e12", "sla", "metrics", solid, "reward + state", "exitX=0;exitY=0.5;entryX=1;entryY=0.5;")
    d.add_edge("e13", "metrics", "training", solid, "team reward")
    d.add_edge("e14", "metrics", "zone_env", solid, "next observation", "exitX=0;exitY=0.5;entryX=0;entryY=0.75;", points=[(290, 431), (290, 265)])
    d.add_edge("e15", "training", "zone_env", dashed, "policy gradients", "exitX=0;exitY=0.5;entryX=0;entryY=0.95;", points=[(285, 578), (285, 325)])
    d.add_edge("e16", "metrics", "dual", solid, "violations", "exitX=1;exitY=0.7;entryX=0;entryY=0.5;")
    d.add_edge("e17", "dual", "training", dashed, "λ feedback", "exitX=0;exitY=0.7;entryX=1;entryY=0.7;")
    return d.to_xml("End-to-End System Architecture")

def build_fig2_legacy():
    d = DrawioDiagram()

    # Muted academic palette. Gray fill distinguishes learned encoders from
    # environment/context boxes without relying on color in print.
    font_base = "fontFamily=Helvetica;fontSize=11;fontColor=#333333;"
    style_actor = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#333333;strokeWidth=1.5;{font_base}align=left;verticalAlign=top;spacingLeft=10;spacingTop=5;fontWeight=bold;"
    style_critic = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#f5f5f5;strokeColor=#333333;strokeWidth=1.2;{font_base}align=center;"
    style_generic = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#555555;strokeWidth=1.2;{font_base}align=center;"
    style_encoder = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#eeeeee;strokeColor=#222222;strokeWidth=1.5;{font_base}align=left;verticalAlign=top;spacingLeft=12;spacingTop=8;"
    style_result = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#f5f5f5;strokeColor=#333333;strokeWidth=1.5;{font_base}align=center;fontWeight=bold;"

    style_region_training = f"rounded=0;fillColor=none;strokeColor=#999999;strokeWidth=1.2;dashed=1;html=1;{font_base}align=right;verticalAlign=top;spacingRight=15;spacingTop=10;"
    style_region_execution = f"rounded=0;fillColor=none;strokeColor=#999999;strokeWidth=1.2;html=1;{font_base}align=right;verticalAlign=top;spacingRight=15;spacingTop=10;"

    d.add_vertex("zone_training", "<b>TRAINING ONLY</b>\n(Centralized value and policy updates)", style_region_training, 50, 40, 720, 245)
    d.add_vertex("zone_execution", "<b>DECENTRALIZED EXECUTION</b>\n(Role-specific observations and causal action chain)", style_region_execution, 50, 320, 720, 760)

    # Centralized training path.
    d.add_vertex("global_state", "Global state s", style_generic, 80, 80, 150, 42)
    d.add_vertex("critic", "Centralized MLP critic\nV(s)", style_critic, 305, 72, 200, 58)
    d.add_vertex("gae", "GAE team advantage", style_critic, 580, 80, 150, 42)
    d.add_vertex("counterfactual", "<b>Actor update</b>\nClipped PPO + action-wise counterfactual target + adaptive entropy", style_critic, 205, 180, 410, 62)

    # Inputs and the graph encoder used by the rack-indexed actors.
    d.add_vertex("rack_history", "Rack history X\n[R x 6 x 7]", style_generic, 85, 385, 150, 55)
    d.add_vertex("role_context", "Role context\nA4: job + network + utilities\nA2: violation utilities", style_generic, 85, 480, 150, 78)

    encoder_text = ("<b>Rack ST-GNN (one instance in A4 and one in A2)</b>\n"
                    "Temporal branch: projection → residual TCN, d = {1, 2, 4}\n"
                    "Spatial branch: current frame → 2 edge-aware GATv2 layers, 4 heads\n"
                    "Fusion: rack embeddings z_r + mean-pooled graph context g\n"
                    "Role context encoder → rack-shared score head")
    d.add_vertex("stgnn", encoder_text, style_encoder, 285, 365, 430, 205)

    a4_text = ("<b>A4 — Placement (ST-GNN)</b>\n"
               "R rack logits + separate DEFER head\n"
               "Masked action: rack 0..R−1 or DEFER")
    d.add_vertex("act_a4", a4_text, style_actor, 105, 640, 270, 82)

    a2_text = ("<b>A2 — Risk localization (ST-GNN)</b>\n"
               "One shared-score logit per rack\n"
               "Mask: positive projected violation")
    d.add_vertex("act_a2", a2_text, style_actor, 445, 640, 270, 82)

    a3_text = ("<b>A3 — Victim selection (MLP)</b>\n"
               "Rack state + up to four candidate slots\n"
               "Mask: positive causal relief")
    d.add_vertex("act_a3", a3_text, style_actor, 445, 790, 270, 82)

    a1_text = ("<b>A1 — Intervention (MLP)</b>\n"
               "Selected rack/victim + relief and cost\n"
               "Action: WAIT or PREEMPT")
    d.add_vertex("act_a1", a1_text, style_actor, 105, 790, 270, 82)

    d.add_vertex("env_step", "Atomic placement/preemption → simulator step", style_result, 250, 985, 320, 45)

    style_edge_solid = "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;endArrow=classic;strokeWidth=1.2;strokeColor=#333333;"
    style_edge_dashed = "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;dashed=1;endArrow=classic;strokeWidth=1.2;strokeColor=#666666;"

    d.add_edge("t1", "global_state", "critic", style_edge_solid)
    d.add_edge("t2", "critic", "gae", style_edge_solid)
    d.add_edge("t3", "gae", "counterfactual", style_edge_solid)
    d.add_edge("t4", "global_state", "counterfactual", style_edge_solid, "counterfactual utilities", "exitX=0.5;exitY=1;entryX=0;entryY=0.5;")

    d.add_edge("g1", "rack_history", "stgnn", style_edge_solid)
    d.add_edge("g2", "role_context", "stgnn", style_edge_solid)
    d.add_edge("g3", "stgnn", "act_a4", style_edge_solid, "A4 logits", "exitX=0.30;exitY=1;entryX=0.5;entryY=0;")
    d.add_edge("g4", "stgnn", "act_a2", style_edge_solid, "A2 logits", "exitX=0.70;exitY=1;entryX=0.5;entryY=0;")

    # Runtime causal chain A4 → A2 → A3 → A1.
    d.add_edge("x1", "act_a4", "act_a2", style_edge_solid, "updated placement", "exitX=1;exitY=0.5;entryX=0;entryY=0.5;")
    d.add_edge("x2", "act_a2", "act_a3", style_edge_solid, "flagged rack")
    d.add_edge("x3", "act_a3", "act_a1", style_edge_solid, "candidate", "exitX=0;exitY=0.5;entryX=1;entryY=0.5;")
    d.add_edge("x4", "act_a1", "env_step", style_edge_solid, "WAIT / PREEMPT")

    # Dashed arrows denote training-only policy updates.
    d.add_edge("up_graph", "counterfactual", "stgnn", style_edge_dashed, "PPO + CF gradients", "exitX=0.65;exitY=1;entryX=0.5;entryY=0;")
    d.add_edge("up_a3", "counterfactual", "act_a3", style_edge_dashed, "", "exitX=0.85;exitY=1;entryX=1;entryY=0.5;", points=[(750, 260), (750, 831)])
    d.add_edge("up_a1", "counterfactual", "act_a1", style_edge_dashed, "", "exitX=0.15;exitY=1;entryX=0;entryY=0.5;", points=[(70, 260), (70, 831)])

    return d.to_xml("ST-GNN Four-Agent CTDE Architecture")


def build_fig2():
    """Landscape view of ST-GNN internals and the four-agent CTDE chain."""
    d = DrawioDiagram()
    font = "fontFamily=Helvetica;fontSize=11;fontColor=#333333;"
    actor = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#333333;strokeWidth=1.5;{font}align=left;verticalAlign=top;spacingLeft=10;spacingTop=7;fontWeight=bold;"
    generic = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#555555;strokeWidth=1.2;{font}align=center;"
    learned = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#eeeeee;strokeColor=#222222;strokeWidth=1.5;{font}align=left;verticalAlign=top;spacingLeft=12;spacingTop=8;"
    zone_train = f"rounded=0;fillColor=none;strokeColor=#999999;strokeWidth=1.2;dashed=1;html=1;{font}align=right;verticalAlign=top;spacingRight=12;spacingTop=8;"
    zone_exec = f"rounded=0;fillColor=none;strokeColor=#999999;strokeWidth=1.2;html=1;{font}align=right;verticalAlign=top;spacingRight=12;spacingTop=8;"
    solid = "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;endArrow=classic;strokeWidth=1.2;strokeColor=#333333;"
    dashed = solid + "dashed=1;strokeColor=#666666;"

    d.add_vertex("train_zone", "<b>TRAINING ONLY</b>", zone_train, 40, 35, 1250, 175)
    d.add_vertex("global", "Global state s", generic, 80, 90, 155, 44)
    d.add_vertex("critic", "Centralized MLP critic V(s)", learned, 300, 82, 210, 60)
    d.add_vertex("gae", "GAE team advantage", learned, 575, 90, 170, 44)
    d.add_vertex("update", "Actor update\nclipped PPO + counterfactual target + adaptive entropy", learned, 830, 75, 360, 72)

    d.add_vertex("exec_zone", "<b>DECENTRALIZED EXECUTION</b>", zone_exec, 40, 250, 1250, 520)
    d.add_vertex("history", "Rack history X\n[R × 6 × 7]", generic, 75, 330, 170, 55)
    d.add_vertex("context", "Role context\nA4: job + network + utilities\nA2: violation utilities", generic, 75, 430, 170, 76)
    stgnn_text = ("<b>Rack ST-GNN</b> — separate instance in A4 and A2\n"
                  "Temporal: projection → residual TCN, d = {1, 2, 4}\n"
                  "Spatial: current frame → 2 edge-aware GATv2 layers, 4 heads\n"
                  "Fusion: rack embeddings z_r and mean-pooled graph context g\n"
                  "Role encoder → rack-shared score head")
    d.add_vertex("stgnn", stgnn_text, learned, 300, 305, 410, 205)

    d.add_vertex("a4", "<b>A4 — Placement (ST-GNN)</b>\nR rack logits + DEFER head\nmask: feasible placement", actor, 770, 300, 220, 78)
    d.add_vertex("a2", "<b>A2 — Risk (ST-GNN)</b>\none logit per rack\nmask: positive violation", actor, 1040, 300, 220, 78)
    d.add_vertex("a3", "<b>A3 — Victim (MLP)</b>\nup to four job slots\nmask: positive relief", actor, 1040, 500, 220, 78)
    d.add_vertex("a1", "<b>A1 — Intervention (MLP)</b>\nrack/victim + relief/cost\nWAIT or PREEMPT", actor, 770, 500, 220, 78)
    d.add_vertex("step", "Atomic action → simulator step", generic, 865, 670, 300, 45)

    d.add_edge("t1", "global", "critic", solid)
    d.add_edge("t2", "critic", "gae", solid)
    d.add_edge("t3", "gae", "update", solid)
    d.add_edge("g1", "history", "stgnn", solid)
    d.add_edge("g2", "context", "stgnn", solid)
    d.add_edge("g3", "stgnn", "a4", solid, "", "exitX=1;exitY=0.35;entryX=0;entryY=0.5;")
    d.add_edge("g4", "stgnn", "a2", solid, "", "exitX=1;exitY=0.7;entryX=0;entryY=0.5;", points=[(735, 448), (1010, 448), (1010, 339)])
    d.add_edge("x1", "a4", "a2", solid, "", "exitX=1;exitY=0.25;entryX=0;entryY=0.25;")
    d.add_edge("x2", "a2", "a3", solid, "flagged rack")
    d.add_edge("x3", "a3", "a1", solid, "", "exitX=0;exitY=0.5;entryX=1;entryY=0.5;")
    d.add_edge("x4", "a1", "step", solid, "WAIT / PREEMPT")

    # Training-only gradients stay above or outside the runtime action chain.
    d.add_edge("u1", "update", "stgnn", dashed, "PPO + CF gradients", "exitX=0.05;exitY=1;entryX=0.5;entryY=0;", points=[(760, 175), (505, 175)])
    d.add_edge("u2", "update", "a3", dashed, "", "exitX=0.95;exitY=1;entryX=1;entryY=0.5;", points=[(1280, 175), (1280, 539)])
    d.add_edge("u3", "update", "a1", dashed, "", "exitX=0.85;exitY=1;entryX=0;entryY=0.5;", points=[(750, 175), (750, 539)])
    return d.to_xml("ST-GNN Four-Agent CTDE Architecture")


def build_fig2_compact():
    """Compact paper figure: ST-GNN encoder and causal runtime actors."""
    d = DrawioDiagram()
    font = "fontFamily=Helvetica;fontSize=11;fontColor=#222222;"
    box = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#444444;strokeWidth=1.2;{font}align=center;"
    learned = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#eeeeee;strokeColor=#222222;strokeWidth=1.4;{font}align=center;fontWeight=bold;"
    agent = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#222222;strokeWidth=1.5;{font}align=center;fontWeight=bold;"
    zone = f"rounded=0;dashed=1;fillColor=none;strokeColor=#999999;strokeWidth=1;html=1;{font}align=right;verticalAlign=top;spacingRight=7;spacingTop=6;"
    solid = "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;endArrow=classic;strokeWidth=1.2;strokeColor=#333333;"

    # Encoder template instantiated separately inside A4 and A2.
    d.add_vertex("enc_zone", "ST-GNN encoder template (separate A4/A2 parameters)", zone, 30, 30, 540, 210)
    d.add_vertex("history", "Rack history\nX ∈ R^(R×6×7)", box, 55, 75, 130, 48)
    d.add_vertex("current", "Current rack graph\nG_t", box, 55, 155, 130, 48)
    d.add_vertex("tcn", "TCN\nd = 1,2,4", learned, 230, 75, 110, 48)
    d.add_vertex("gat", "2× GATv2\n4 heads", learned, 230, 155, 110, 48)
    d.add_vertex("fusion", "Fusion\nz_r , g", learned, 395, 112, 120, 56)
    d.add_edge("e_hist", "history", "tcn", solid)
    d.add_edge("e_cur", "current", "gat", solid)
    d.add_edge("e_tcn", "tcn", "fusion", solid)
    d.add_edge("e_gat", "gat", "fusion", solid)

    # Runtime causal chain.
    d.add_vertex("a4", "A4  Placement\nST-GNN + DEFER", agent, 625, 45, 150, 58)
    d.add_vertex("a2", "A2  Risk\nST-GNN", agent, 845, 45, 150, 58)
    d.add_vertex("a3", "A3  Victim\nMLP", agent, 845, 165, 150, 58)
    d.add_vertex("a1", "A1  Intervention\nMLP", agent, 625, 165, 150, 58)
    d.add_vertex("step", "Simulator step", box, 1060, 105, 135, 58)
    d.add_edge("e_fuse_a4", "fusion", "a4", solid)
    d.add_edge("e_a4_a2", "a4", "a2", solid)
    d.add_edge("e_a2_a3", "a2", "a3", solid)
    d.add_edge("e_a3_a1", "a3", "a1", solid, "", "exitX=0;exitY=0.5;entryX=1;entryY=0.5;")
    d.add_edge("e_a1_step", "a1", "step", solid, "", "exitX=0.5;exitY=1;entryX=0;entryY=0.75;", points=[(700, 250), (1025, 250), (1025, 149)])

    return d.to_xml("ST-GNN Encoder and Four-Agent Runtime")


def build_fig3():
    d = DrawioDiagram()
    
    # Styles - Clean neutral styling for nodes. Color used ONLY for meaningful flow/logical grouping.
    font_base = "fontFamily=Helvetica;fontSize=10;fontColor=#333333;"
    style_sw = f"shape=cube;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;fillColor=#f5f5f5;strokeColor=#333333;strokeWidth=1.2;html=1;{font_base}align=center;"
    style_host_regular = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#ffffff;strokeColor=#cccccc;strokeWidth=1.2;html=1;{font_base}align=center;"
    
    # Semantic Colors: Green border for green flow hosts, Orange border for orange flow hosts
    style_host_gang_green = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#e2f0d9;strokeColor=#385723;strokeWidth=2;html=1;{font_base}align=center;fontWeight=bold;"
    style_host_gang_orange = f"rounded=1;whiteSpace=wrap;html=1;fillColor=#fce4d6;strokeColor=#c65911;strokeWidth=2;html=1;{font_base}align=center;fontWeight=bold;"
    
    style_outline = f"rounded=0;fillColor=none;strokeColor=#a0a0a0;strokeWidth=1.5;dashed=1;html=1;{font_base}align=left;verticalAlign=top;spacingLeft=10;spacingTop=10;"

    # 1. Network Topology drawing (Height reduced to 230 to keep it compact since there is no legend inside)
    d.add_vertex("topo_zone", "", style_outline, 50, 40, 720, 230)

    # Switches - Edge Switches Y shifted up to 125. Height increased to 65 to fit "Edge 0\n(Rack 0)" vertically!
    # Width set to 120 so it looks properly proportioned.
    d.add_vertex("sw_agg", "Aggregate\nSwitch", style_sw, 350, 60, 120, 50)
    
    d.add_vertex("sw_edge0", "Edge 0\n(Rack 0)", style_sw, 145, 125, 120, 65)
    d.add_vertex("sw_edge1", "Edge 1\n(Rack 1)", style_sw, 375, 125, 120, 65)
    d.add_vertex("sw_edge2", "Edge 2\n(Rack 2)", style_sw, 605, 125, 120, 65)

    # Hosts (highlighting Gang J = {H0, H1, H4, H5}) (Centered properly with increased spacing to 30px)
    # Rack 0 (Edge 0)
    d.add_vertex("h0", "H0\n(Gang J)", style_host_gang_green, 100, 220, 50, 35)
    d.add_vertex("h1", "H1\n(Gang J)", style_host_gang_green, 180, 220, 50, 35)
    d.add_vertex("h2", "H2", style_host_regular, 260, 220, 50, 35)

    # Rack 1 (Edge 1)
    d.add_vertex("h4", "H4\n(Gang J)", style_host_gang_orange, 330, 220, 50, 35)
    d.add_vertex("h5", "H5\n(Gang J)", style_host_gang_orange, 410, 220, 50, 35)
    d.add_vertex("h6", "H6", style_host_regular, 490, 220, 50, 35)

    # Rack 2 (Edge 2)
    d.add_vertex("h8", "H8", style_host_regular, 560, 220, 50, 35)
    d.add_vertex("h9", "H9", style_host_regular, 640, 220, 50, 35)
    d.add_vertex("h10", "H10", style_host_regular, 720, 220, 50, 35)

    # Connections for Fat-Tree topology
    style_link = "edgeStyle=orthogonalEdgeStyle;rounded=0;endArrow=none;strokeColor=#888888;strokeWidth=1.5;"
    
    # Agg to Edges
    d.add_edge("l_agg_e0", "sw_agg", "sw_edge0", style_link)
    d.add_edge("l_agg_e1", "sw_agg", "sw_edge1", style_link)
    d.add_edge("l_agg_e2", "sw_agg", "sw_edge2", style_link)

    # Edges to Hosts (Explicit top center entries)
    d.add_edge("l_e0_h0", "sw_edge0", "h0", style_link, entry_exit="entryX=0.5;entryY=0;")
    d.add_edge("l_e0_h1", "sw_edge0", "h1", style_link, entry_exit="entryX=0.5;entryY=0;")
    d.add_edge("l_e0_h2", "sw_edge0", "h2", style_link, entry_exit="entryX=0.5;entryY=0;")

    d.add_edge("l_e1_h4", "sw_edge1", "h4", style_link, entry_exit="entryX=0.5;entryY=0;")
    d.add_edge("l_e1_h5", "sw_edge1", "h5", style_link, entry_exit="entryX=0.5;entryY=0;")
    d.add_edge("l_e1_h6", "sw_edge1", "h6", style_link, entry_exit="entryX=0.5;entryY=0;")

    d.add_edge("l_e2_h8", "sw_edge2", "h8", style_link, entry_exit="entryX=0.5;entryY=0;")
    d.add_edge("l_e2_h9", "sw_edge2", "h9", style_link, entry_exit="entryX=0.5;entryY=0;")
    d.add_edge("l_e2_h10", "sw_edge2", "h10", style_link, entry_exit="entryX=0.5;entryY=0;")

    # Ring All-Reduce Path arrows (Thick colored links)
    style_intra = "edgeStyle=orthogonalEdgeStyle;rounded=1;endArrow=classic;strokeColor=#2ca02c;strokeWidth=3;html=1;"
    style_cross = "edgeStyle=orthogonalEdgeStyle;rounded=1;endArrow=classic;strokeColor=#ff7f0e;strokeWidth=3;html=1;"

    # 1. H0 -> H1 (Intra-rack, horizontal center connect)
    d.add_edge("ring_h0_h1", "h0", "h1", style_intra, "", "exitX=1;exitY=0.5;entryX=0;entryY=0.5;")
    
    # 2. H1 -> H4 (Cross-rack, routed cleanly through the gap, goes UP first to avoid crossing H2!)
    d.add_edge("ring_h1_h4", "h1", "h4", style_cross, "cross-rack flow", "exitX=0.8;exitY=0;entryX=0.2;entryY=0;",
               points=[(220, 200), (320, 200), (320, 110), (340, 110)])
               
    # 3. H4 -> H5 (Intra-rack, horizontal center connect)
    d.add_edge("ring_h4_h5", "h4", "h5", style_intra, "", "exitX=1;exitY=0.5;entryX=0;entryY=0.5;")
    
    # 4. H5 -> H0 (Cross-rack, routed cleanly around on the far right and above switches, goes UP first to avoid crossing H6!)
    d.add_edge("ring_h5_h0", "h5", "h0", style_cross, "cross-rack flow", "exitX=0.8;exitY=0;entryX=0.2;entryY=0;",
               points=[(450, 200), (550, 200), (550, 50), (110, 50)])

    return d.to_xml("Network Topology")


def find_drawio_binary():
    """Locate the diagrams.net desktop exporter without requiring installation."""
    candidates = [
        os.environ.get("DRAWIO_BIN"),
        shutil.which("drawio"),
        shutil.which("draw.io"),
        "/Applications/draw.io-M1.app/Contents/MacOS/draw.io",
        "/Applications/draw.io.app/Contents/MacOS/draw.io",
    ]
    return next((path for path in candidates if path and os.path.isfile(path)), None)


def export_pdfs():
    """Export cropped PDFs to the paths consumed by main.tex."""
    drawio = find_drawio_binary()
    if drawio is None:
        print("draw.io desktop exporter not found; generated editable .drawio files only")
        return
    os.makedirs("images", exist_ok=True)
    outputs = {
        "figures/fig1_sys_arch.drawio": "images/system_architecture_workloads.pdf",
        "figures/fig2_marl_ctde.drawio": "images/marl_stgnn_runtime_compact.pdf",
        "figures/fig3_network_topology.drawio": "images/ring_network_topology.pdf",
    }
    for source, target in outputs.items():
        subprocess.run(
            [drawio, "-x", "-f", "pdf", "--crop", "-b", "10",
             "-o", target, source],
            check=True,
        )
        # This local draw.io build appends a blank second page to CLI PDF
        # exports. Keep only the diagram page so graphicx sees a one-page asset.
        qpdf = shutil.which("qpdf")
        if qpdf:
            trimmed = target + ".trimmed.pdf"
            subprocess.run(
                [qpdf, target, "--pages", target, "1", "--", trimmed],
                check=True,
            )
            os.replace(trimmed, target)
        print(f"Exported {target}")

def main():
    os.makedirs('figures', exist_ok=True)
    
    fig1_xml = build_fig1()
    with open('figures/fig1_sys_arch.drawio', 'w') as f:
        f.write(fig1_xml)
    print("Generated figures/fig1_sys_arch.drawio")

    fig2_xml = build_fig2_compact()
    with open('figures/fig2_marl_ctde.drawio', 'w') as f:
        f.write(fig2_xml)
    print("Generated figures/fig2_marl_ctde.drawio")

    fig3_xml = build_fig3()
    with open('figures/fig3_network_topology.drawio', 'w') as f:
        f.write(fig3_xml)
    print("Generated figures/fig3_network_topology.drawio")

    export_pdfs()

if __name__ == '__main__':
    main()
