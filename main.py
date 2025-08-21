from utils.my_logger import init_logger   

from packaging import version
import pydantic

assert version.parse(pydantic.VERSION) >= version.parse("2.0.0"), "require: pydantic.VERSION >= v2"

from model import GPTModel, GPTOption, ModelIn , BaseModel

from enum import Enum
from typing import List, Optional
import random

class Identity(Enum):
    CIVILIAN = "平民"
    WOLF = "狼人" 
    BOMBER = "炸彈魔"
    
class GameResult(Enum):
    CIVILIAN_WIN = "平民獲勝"
    WOLF_WIN = "狼人獲勝"
    BOMBER_WIN = "炸彈魔獲勝"
    
class IdentityCard:
    def __init__(self, identity: Identity):
        self.identity = identity
    
    def __str__(self):
        return self.identity.value
    
    def get_identity(self):
        return self.identity

class Player:
    """AI玩家類別"""
    def __init__(self, player_name: str, player_number: int, model: BaseModel):
        self.player_name = player_name
        self.player_number = player_number
        self.identity_card = IdentityCard(None)
        self.model = model
        #+型別檢查
        self.conversation_history = []  # 記住對話歷史
    
    def set_identity_card(self, card: IdentityCard):
        """設定玩家身分卡"""
        self.identity_card = card
    
    def get_identity_info(self) -> str:
        """return玩家身分卡資訊"""
        return f"{self.identity_card}"
    
    def choose_card_ai(self, card1: IdentityCard, card2: IdentityCard = None) -> IdentityCard:
        """ 詢問玩家要選哪張身分(AI)，變更身分，return傳給下一位的卡 """
        if card2 is None:
            # 如果只有一張卡，直接設為自己的身分
            self.set_identity_card(card1)
            return None
        else:
            # 使用 GPT 來決定選擇哪張卡
            system_prompt = f"""你是狼人殺遊戲的玩家{self.player_number}({self.player_name})。
你需要在兩張身分卡中選擇一張作為自己的身分，另一張傳給下一位玩家。

遊戲規則：
- 狼人：需要隱藏身分，讓平民互相猜疑
- 炸彈魔：如果被票出就獲勝，所以可能想要引起懷疑
- 平民：需要找出狼人，避免被票出

請根據策略思考選擇哪張卡片。"""

            prompt = f"""你現在有兩張卡片可以選擇：
卡片1: {card1}
卡片2: {card2}

請選擇其中一張作為你的身分卡，另一張將傳給下一位玩家。
請只回答 "1" 或 "2" 來表示你的選擇。"""

            message = ModelIn(
                content=prompt,
                system_prompt=system_prompt,
                #+thinking問題
                thinking=False
            )
            
            response = self.gpt_model.chat(message)
            choice = response.get('output', '1').strip()
            #+少存資料
            # 解析選擇
            if '2' in choice:
                self.set_identity_card(card2)
                return card1
            else:
                self.set_identity_card(card1)
                return card2
    
    def declare_card_ai(self, card_to_pass: IdentityCard) -> str:
        """ 宣告要傳給下一位玩家的卡片 """
        system_prompt = f"""你是狼人殺遊戲的玩家{self.player_number}({self.player_name})。
你的真實身分是：{self.identity_card}
你要傳給下一位玩家的卡片是：{card_to_pass}

你可以選擇說實話或撒謊：
- 如果你是狼人：可能想要撒謊來誤導其他玩家
- 如果你是炸彈魔：可能想要製造混亂
- 如果你是平民：通常會說實話，但也可能為了策略而撒謊

請宣告你要傳給下一位玩家的卡片。"""

        prompt = f"""你要傳給下一位玩家的卡片是：{card_to_pass}
請宣告這張卡片（可以說實話或撒謊，根據你的策略）。
請用自然的語言回答，例如："我傳給下一位的是平民" 或 "我要傳狼人卡"等。"""

        message = ModelIn(
            content=prompt,
            system_prompt=system_prompt,
            #+thinking問題
            thinking=False
        )
        
        response = self.gpt_model.chat(message)
        declaration = f"{self.player_name}: {response.get('output', '我傳給下一位的是平民').strip()}"
        self.conversation_history.append(declaration)
        return declaration
    
    def declare_discard_ai(self, card_to_discard: IdentityCard) -> str:
        """ 最後一位玩家宣告要丟棄的卡片 """
        system_prompt = f"""你是狼人殺遊戲的玩家{self.player_number}({self.player_name})。
你的真實身分是：{self.identity_card}
你是最後一位玩家，需要宣告你要丢棄的卡片：{card_to_discard}

作為最後一位玩家，你的宣告很重要，因為：
- 如果你是狼人：可能想要撒謊來誤導其他玩家，讓他們以為丟棄的是其他卡片
- 如果你是炸彈魔：可能想要製造混亂或讓別人懷疑
- 如果你是平民：通常會說實話，但也可能為了策略而撒謊

請宣告你要丟棄的卡片。"""

        prompt = f"""你要丟棄的卡片是：{card_to_discard}
作為最後一位玩家，請宣告這張丟棄的卡片（可以說實話或撒謊，根據你的策略）。
請用自然的語言回答，例如："我丟棄的是平民" 或 "我要丟掉狼人卡"等。"""

        message = ModelIn(
            content=prompt,
            system_prompt=system_prompt,
            #+thinking問題
            thinking=False
        )
        
        response = self.gpt_model.chat(message)
        declaration = f"{self.player_name}: {response.get('output', '我丟棄的是平民').strip()}"
        self.conversation_history.append(declaration)
        return declaration
    
    def think_about_conversation_ai(self, conversation: str) -> str:
        """ input對話給玩家思考(AI)，return玩家看法的對話 """
        system_prompt = f"""你是狼人殺遊戲的玩家{self.player_number}({self.player_name})。
你的真實身分是：{self.identity_card}

你需要分析其他玩家的宣告，並給出你的看法。
根據你的身分：
- 如果你是狼人：試圖誤導其他人，讓平民互相懷疑
- 如果你是炸彈魔：可能想要引起別人的懷疑（因為被票出就獲勝）
- 如果你是平民：試圖找出真正的狼人

過往對話歷史：
{chr(10).join(self.conversation_history)}"""

        prompt = f"""剛才聽到這句話："{conversation}"
請分析這個宣告，並給出你的看法和推理。
請用自然語言表達你的想法，例如分析對方是否可信、是否有疑點等。"""

        message = ModelIn(
            content=prompt,
            system_prompt=system_prompt,
            #+thinking問題
            thinking=False
        )
        
        response = self.gpt_model.chat(message)
        thought = f"{self.player_name}: {response.get('output', '我覺得這個宣告很可疑').strip()}"
        self.conversation_history.append(thought)
        return thought
    
    def vote_for_werewolf_ai(self, all_conversations: List[str]) -> int:
        """ input對話給玩家思考(AI)，return玩家認為哪位玩家是狼 """
        system_prompt = f"""你是狼人殺遊戲的玩家{self.player_number}({self.player_name})。
你的真實身分是：{self.identity_card}

現在是投票階段，你需要投票選出一個玩家。
根據你的身分：
- 如果你是狼人：避免投票給炸彈魔（因為炸彈魔被票出會獲勝），優先投票給平民
- 如果你是炸彈魔：可能想要讓別人投票給你（因為你被票出就獲勝），但不要太明顯
- 如果你是平民：試圖投票給狼人

遊戲中有玩家1到6，你不能投票給自己（你是玩家{self.player_number}）。"""

        all_conv_text = "\n".join(all_conversations)
        prompt = f"""根據以下所有的對話內容，請分析並決定投票給哪位玩家：

{all_conv_text}

請仔細分析每位玩家的言論，找出最可疑的人。
請只回答一個數字（1-6），代表你要投票的玩家號碼。
記住你不能投票給自己（玩家{self.player_number}）。"""

        message = ModelIn(
            content=prompt,
            system_prompt=system_prompt,
            #+thinking問題
            thinking=False
        )
        
        response = self.gpt_model.chat(message)
        vote_text = response.get('output', '1').strip()
        
        # 解析投票結果
        try:
            vote = int(''.join(filter(str.isdigit, vote_text))[:1])  # 取第一個數字
            if vote < 1 or vote > 6 or vote == self.player_number:
                # 如果投票無效，隨機選擇
                available_targets = [i for i in range(1, 7) if i != self.player_number]
                vote = random.choice(available_targets)
        except:
            available_targets = [i for i in range(1, 7) if i != self.player_number]
            vote = random.choice(available_targets)
            
        return vote

class CardDeck:
    def __init__(self):
        self.cards: List[IdentityCard] = []
        self._initialize_deck()
    
    def _initialize_deck(self):
        self.cards.append(IdentityCard(Identity.WOLF))
        self.cards.append(IdentityCard(Identity.BOMBER))
        for _ in range(5): 
            self.cards.append(IdentityCard(Identity.CIVILIAN))
        random.shuffle(self.cards)
    
    
    def draw_card(self) -> Optional[IdentityCard]:
        return self.cards.pop() if self.cards else None
    
    def get_deck(self):
        return self.cards
    
    def update_deck(self, new_deck):
        self.cards = new_deck

class GameFlow:
    """遊戲流程控制"""
    def __init__(self, players: List[Player], card_deck: CardDeck):
        self.players = players
        self.card_deck = card_deck
    
    def explain_rules(self):
        """解釋遊戲規則"""
        print("=== AI 狼人殺遊戲規則 ===")
        print("1. 每位 AI 玩家會拿到身分卡")
        print("2. AI 玩家需要宣告傳給下一位的卡片（可能撒謊）")
        print("3. AI 玩家會討論並分析其他人的言論")
        print("4. 最後 AI 玩家投票找出狼人")
        print("5. 如果票出炸彈魔，炸彈魔獲勝")
        print("6. 如果票出平民，狼人獲勝")
        print("7. 如果票出狼人，平民獲勝")
    
    def process_turn(self, player: Player, previous_card: Optional[IdentityCard]) -> IdentityCard:
        """處理玩家回合"""
        if previous_card is None:
            # 第一個玩家，抽兩張卡
            card1 = self.card_deck.draw_card()
            card2 = self.card_deck.draw_card()
            return player.choose_card_ai(card1, card2)
        else:
            # 其他玩家，拿到上一位傳來的卡和新抽的卡
            new_card = self.card_deck.draw_card()
            return player.choose_card_ai(previous_card, new_card)
    
    def get_declaration(self, player: Player, card_to_pass: IdentityCard) -> str:
        """獲取玩家宣告"""
        return player.declare_card_ai(card_to_pass)
    
    def get_player_thoughts(self, player: Player, conversation: str) -> str:
        """獲取玩家對對話的想法"""
        return player.think_about_conversation_ai(conversation)
    
    def get_vote(self, player: Player, all_conversations: List[str]) -> int:
        """獲取玩家投票"""
        return player.vote_for_werewolf_ai(all_conversations)

# 主程式
def main():
    # 創建 GPT 模型配置
    gpt_option = GPTOption(
        model="gpt-4o",  # 使用較好的模型進行推理
        temperature=0.8,  # 適中的創造性
        max_output_tokens=1024,
        stream=False  # 不使用串流模式以便程式處理
    )
    
    # 創建玩家，每個玩家都有自己的 GPT 模型實例
    player_names = ["Alice", "Bob", "Charlie", "David", "Emily", "Frank"]
    players = []
    
    print("=== 初始化 AI 玩家 ===")
    for i, name in enumerate(player_names):
        gpt_model = GPTModel(opt=gpt_option)
        player = Player(name, i + 1, gpt_model)
        players.append(player)
        print(f"AI 玩家 {i+1} ({name}) 初始化完成")

    # 創建卡牌堆
    card_deck = CardDeck()

    # 創建遊戲流程控制器
    game_flow = GameFlow(players, card_deck)
    print("\n=== AI 狼人殺遊戲開始 ===")
    game_flow.explain_rules()

    # 抽卡與宣告階段
    print("\n=== AI 抽卡與宣告階段 ===")
    conversations = []
    previous_card = None

    for i, player in enumerate(players):
        print(f"\n>>> 輪到 AI 玩家 {player.player_name} <<<")
        
        # 處理回合
        card_to_pass = game_flow.process_turn(player, previous_card)
        
        # 顯示玩家獲得的身分
        print(f"{player.player_name} 獲得身分：{player.identity_card}")
        
        # 所有玩家都要宣告要傳出/丟棄的卡片
        if card_to_pass:
            print(f"{player.player_name} 正在思考如何宣告...")
            if i < len(players) - 1:
                # 不是最後一個玩家：宣告傳給下一位的卡片
                declaration = game_flow.get_declaration(player, card_to_pass)
                conversations.append(declaration)
                print(f"宣告：{declaration}")
                previous_card = card_to_pass
            else:
                # 最後一個玩家：宣告丟棄的卡片
                discard_declaration = player.declare_discard_ai(card_to_pass)
                conversations.append(discard_declaration)
                print(f"丟棄宣告：{discard_declaration}")
                
                # 最後一個玩家的卡放回卡牌堆
                current_deck = card_deck.get_deck()
                current_deck.append(card_to_pass)
                card_deck.update_deck(current_deck)
        
    # 討論階段
    print("\n=== AI 討論階段 ===")
    discussion_thoughts = []
    
    for player in players:
        print(f"\n>>> {player.player_name} 的分析 <<<")
        for conversation in conversations:
            if not conversation.startswith(player.player_name):  # 不評論自己的話
                print(f"{player.player_name} 正在分析: {conversation}")
                thought = game_flow.get_player_thoughts(player, conversation)
                discussion_thoughts.append(thought)
                print(f"想法：{thought}")
        
    # 投票階段
    print("\n=== AI 投票階段 ===")
    votes = {}
    all_conversations = conversations + discussion_thoughts

    for player in players:
        print(f"\n>>> {player.player_name} 正在決定投票 <<<")
        vote_target = game_flow.get_vote(player, all_conversations)
        votes[player.player_number] = vote_target
        print(f"{player.player_name} 投票給 玩家{vote_target}")

    # 計算結果
    print("\n=== 遊戲結果 ===")
    
    # 選出最高票的玩家
    vote_counts = {}
    for vote_target in votes.values():
        vote_counts[vote_target] = vote_counts.get(vote_target, 0) + 1

    # 找出得票最高的玩家
    eliminated_player_number = max(vote_counts.keys(), key=lambda x: vote_counts[x])
    eliminated_player = players[eliminated_player_number - 1]

    print(f"\n投票詳情：")
    for player_num, target in votes.items():
        voter_name = players[player_num - 1].player_name
        target_name = players[target - 1].player_name
        print(f"{voter_name} (玩家{player_num}) → {target_name} (玩家{target})")

    print(f"\n {eliminated_player.player_name} 被 AI 們票出局！")
    print(f" {eliminated_player.player_name} 的真實身分是：{eliminated_player.identity_card}")

    # 判定勝負
    eliminated_identity = eliminated_player.identity_card.get_identity()

    print(f"\n遊戲結果：")
    if eliminated_identity == Identity.BOMBER:
        print("炸彈魔被票出，炸彈魔獲勝！")
    elif eliminated_identity == Identity.CIVILIAN:
        print("平民被票出，狼人獲勝！")
    elif eliminated_identity == Identity.WOLF:
        print("狼人被票出，平民獲勝！")

    # 顯示所有玩家的最終身分
    print("\n=== 所有 AI 玩家身分揭曉 ===")
    for player in players:
        print(f" {player.get_identity_info()}")

if __name__ == "__main__":
    main()