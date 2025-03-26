from flask import Flask, jsonify, render_template
import requests

# Constants
JITA_REGION_ID = 10000002  # The Forge region
PLEX_TYPE_ID = 44992       # PLEX type ID

app = Flask(__name__)

def fetch_market_orders():
    url = f'https://esi.evetech.net/latest/markets/{JITA_REGION_ID}/orders/?type_id={PLEX_TYPE_ID}'
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def calculate_prices(orders):
    buy_orders = [order for order in orders if order['is_buy_order']]
    sell_orders = [order for order in orders if not order['is_buy_order']]

    highest_buy_order = max(buy_orders, key=lambda x: x['price']) if buy_orders else None
    lowest_sell_order = min(sell_orders, key=lambda x: x['price']) if sell_orders else None

    return {
        "highest_buy": {
            "price": highest_buy_order['price'] if highest_buy_order else None,
            "volume": highest_buy_order['volume_remain'] if highest_buy_order else None
        },
        "lowest_sell": {
            "price": lowest_sell_order['price'] if lowest_sell_order else None,
            "volume": lowest_sell_order['volume_remain'] if lowest_sell_order else None
        }
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/plex-price', methods=['GET'])
def get_plex_price():
    try:
        orders = fetch_market_orders()
        data = calculate_prices(orders)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# This runs only if started directly (not imported)
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)


#get them as dictionary.
#data strucutre: {'duration': 30, 'is_buy_order': False, 'issued': '2025-03-26T21:58:08Z', 'location_id': 60003760, 'min_volume': 1, 'order_id': 7017076655, 'price': 6180000.0, 'range': 'region', 'system_id': 30000142, 'type_id': 44992, 'volume_remain': 2095, 'volume_total': 3000}